#include "Scheduler.h"

#include "Metrics.h"

#include <chrono>
#include <thread>
#include <algorithm>  
#include <condition_variable>
#include <mutex>
#include <list>
#include <malloc.h>  

#include "Utility\Settings.h"
#include "GPGPU\GPUManager.h"

//Scheduler Variables//
static std::thread* threads;
static std::size_t threadCount;
static boost::fibers::condition_variable_any threadCondition;


namespace Scheduler {

	namespace detail {

		bool shouldRun;
		bool needsSort;

		std::thread::id mainID;

		std::size_t fiberCount;
		std::mutex fiberMutex;

		boost::fibers::fiber_specific_ptr<std::size_t>* holdCount;
		boost::fibers::fiber_specific_ptr<std::mutex>* holdMutex;
		boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable_any>* blockCondition;

		//clean up the block datatype (needs 64 alignment)
		void CleanUpMutex(std::mutex* ptr) {
			ptr->~mutex();
			delete ptr;
		}

		//clean up the block datatype (needs 64 alignment)
		void CleanUpAlignedCondition(boost::fibers::condition_variable_any* ptr) {
			ptr->~condition_variable_any();

			//TODO: Make this aligned_alloc with c++17, not visual studio specific code
			_aligned_free(ptr);
		}

		//Init the fiber specific stuff
		void InitPointers() {
			if (!detail::holdMutex->get()) {
				detail::holdMutex->reset(new std::mutex);
			}
			if (!detail::holdCount->get()) {
				detail::holdCount->reset(new std::size_t(0));
			}
			if (!detail::blockCondition->get()) {

				//TODO: Make this aligned_alloc with c++17, not visual studio specific code
				boost::fibers::condition_variable_any* newData =
					(boost::fibers::condition_variable_any*)_aligned_malloc(sizeof(boost::fibers::condition_variable_any), 64); //needs 64 alignment
				new (newData) boost::fibers::condition_variable_any();
				detail::blockCondition->reset(newData);
			}
		}

		class SoulScheduler :
			public boost::fibers::algo::algorithm_with_properties< FiberProperties > {
		private:
			typedef std::list< boost::fibers::context * >  rqueue_t;
			typedef boost::fibers::scheduler::ready_queue_t lqueue_t;

			static rqueue_t     	readyQueue;
			static std::mutex   	queueMutex;
			static rqueue_t     	mainOnlyQueue;

			lqueue_t            	localQueue{};
			std::mutex              mtx_{};
			std::condition_variable cnd_{};
			bool                    flag_{ false };
			bool                    suspend_;

		public:
			SoulScheduler() = default;

			SoulScheduler(bool suspend) :
				suspend_{ suspend } {
			}

			SoulScheduler(SoulScheduler const&) = delete;
			SoulScheduler(SoulScheduler &&) = delete;

			SoulScheduler & operator=(SoulScheduler const&) = delete;
			SoulScheduler & operator=(SoulScheduler &&) = delete;

			void InsertContext(rqueue_t& queue, boost::fibers::context*& ctx, int ctxPriority) {
				rqueue_t::iterator i(std::find_if(queue.begin(), queue.end(),
					[ctxPriority, this](boost::fibers::context* c)
				{ return properties(c).GetPriority() < ctxPriority; }));

				queue.insert(i, ctx);
			}

			virtual void awakened(boost::fibers::context * ctx, FiberProperties& props) noexcept {

				//dont push fiber when helper or the main fiber is passed in
				if (ctx->is_context(boost::fibers::type::pinned_context)) {
					localQueue.push_back(*ctx);
				}
				else {
					ctx->detach();

					std::unique_lock< std::mutex > lk(queueMutex);

					int ctxPriority = props.GetPriority();

					//if it needs to run on the main thread
					if (props.RunOnMain()) {
						InsertContext(mainOnlyQueue, ctx, ctxPriority);
					}
					else {
						InsertContext(readyQueue, ctx, ctxPriority);
					}
				}
			}

			virtual boost::fibers::context * pick_next() noexcept {

				boost::fibers::context * ctx(nullptr);
				std::thread::id thisID = std::this_thread::get_id();


				std::unique_lock< std::mutex > lk(queueMutex);

				if (!mainOnlyQueue.empty() && thisID == mainID) {
					ctx = mainOnlyQueue.front();
					mainOnlyQueue.pop_front();
					lk.unlock();
					BOOST_ASSERT(nullptr != ctx);
					boost::fibers::context::active()->attach(ctx);
				}
				else if (!readyQueue.empty()) {
					ctx = readyQueue.front();
					readyQueue.pop_front();
					lk.unlock();
					BOOST_ASSERT(nullptr != ctx);
					boost::fibers::context::active()->attach(ctx);
				}
				else {
					lk.unlock();

					if (!localQueue.empty()) {
						ctx = &localQueue.front();
						localQueue.pop_front();
					}
				}
				return ctx;
			}

			virtual bool has_ready_fibers() const noexcept {
				std::unique_lock< std::mutex > lock(queueMutex);
				return !mainOnlyQueue.empty() || !readyQueue.empty() || !localQueue.empty();
			}

			virtual void property_change(boost::fibers::context * ctx, FiberProperties & props) noexcept {
				if (!ctx->ready_is_linked()) {
					return;
				}
				ctx->ready_unlink();
				awakened(ctx, props);
			}

			void suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept {
				if (suspend_) {
					if ((std::chrono::steady_clock::time_point::max)() == time_point) {
						std::unique_lock< std::mutex > lk(mtx_);
						cnd_.wait(lk, [this]() { return flag_; });
						flag_ = false;
					}
					else {
						std::unique_lock< std::mutex > lk(mtx_);
						cnd_.wait_until(lk, time_point, [this]() { return flag_; });
						flag_ = false;
					}
				}
			}
			void notify() noexcept {
				if (suspend_) {
					std::unique_lock< std::mutex > lk(mtx_);
					flag_ = true;
					lk.unlock();
					cnd_.notify_all();
				}
			}

		};

		SoulScheduler::rqueue_t SoulScheduler::readyQueue{};
		SoulScheduler::rqueue_t SoulScheduler::mainOnlyQueue{};
		std::mutex SoulScheduler::queueMutex{};


		//launches a thread that waits with a fiber conditional, meaning it still executes fibers despite waiting for a notify release
		void ThreadRun() {
			boost::fibers::use_scheduling_algorithm<SoulScheduler >();

			GPUManager::InitThread();

			std::unique_lock<std::mutex> lock(Scheduler::detail::fiberMutex);
			threadCondition.wait(lock, []() { return 0 == Scheduler::detail::fiberCount && !detail::shouldRun; });
		}

	}

	void Terminate() {

		detail::fiberMutex.lock();
		detail::shouldRun = false;
		--detail::fiberCount;
		detail::fiberMutex.unlock();

#ifndef	SOUL_SINGLE_STACK

		//yield this fiber until all the remaining work is done

		bool run = true;
		while (run) {
			detail::fiberMutex.lock();
			if (detail::fiberCount == 0) {
				run = false;
				detail::fiberMutex.unlock();
				threadCondition.notify_all();
			}
			else {
				detail::fiberMutex.unlock();
				boost::this_fiber::yield();
			}
		}

		//join all complete threads
		for (uint i = 0; i < threadCount; ++i) {
			threads[i].join();
		}

		delete[] threads;

		delete detail::holdCount;
		delete detail::holdMutex;
		delete detail::blockCondition;

#endif

	}

	void Init() {


		threadCount = 0;
		detail::fiberCount = 0;
		detail::shouldRun = true;
		detail::needsSort = false;

#ifndef	SOUL_SINGLE_STACK

		detail::holdCount = new boost::fibers::fiber_specific_ptr<std::size_t>;
		detail::holdMutex = new boost::fibers::fiber_specific_ptr<std::mutex>(detail::CleanUpMutex);
		detail::blockCondition =
			new boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable_any>(detail::CleanUpAlignedCondition);

		detail::mainID = std::this_thread::get_id();

		boost::fibers::use_scheduling_algorithm< detail::SoulScheduler >();

		//the main thread takes up one slot, leave one open for system+background.
		Settings::Get("Engine.Additional_Threads",size_t(std::thread::hardware_concurrency()-1), &threadCount);
		threads = new std::thread[threadCount];

		detail::fiberCount++;

		for (uint i = 0; i < threadCount; ++i) {
			threads[i] = std::thread(detail::ThreadRun);
		}

		//init the main fiber specifics
		detail::InitPointers();

#endif

	}


	void Block() {

#ifndef	SOUL_SINGLE_STACK


		//get the current fibers stats for blocking
		std::size_t* holdSize = detail::holdCount->get();

		std::unique_lock<std::mutex> lock(*detail::holdMutex->get());
		(*detail::blockCondition)->wait(lock, [=]() { return 0 == *holdSize; });

		assert(*holdSize == 0);

#endif


	}

	void Defer() {

#ifndef	SOUL_SINGLE_STACK

		boost::this_fiber::yield();

#endif

	}

	bool Running() {

		return detail::shouldRun;
	}

};
