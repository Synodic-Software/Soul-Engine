#include "Scheduler.h"

#include "Metrics.h"

#include <thread>
#include <algorithm>  
#include <condition_variable>
#include <mutex>
#include <deque>

//Scheduler Variables//
static std::thread* threads;
static std::size_t threadCount{ 0 };
static boost::fibers::condition_variable_any threadCondition{};



namespace Scheduler {

	namespace detail {

		bool shouldRun = true;

		std::thread::id mainID;

		std::size_t fiberCount = 0;
		std::mutex fiberMutex;

		boost::fibers::fiber_specific_ptr<std::size_t> holdCount;
		boost::fibers::fiber_specific_ptr<std::mutex> holdMutex;
		boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable_any> blockCondition;

		void InitCheck() {
			if (!detail::holdMutex.get()) {
				detail::holdMutex.reset(new std::mutex);
			}

			if (!detail::holdCount.get()) {
				detail::holdCount.reset(new std::size_t(0));
			}

			if (!detail::blockCondition.get()) {
				detail::blockCondition.reset(new boost::fibers::condition_variable_any);
			}
		}

		class SoulScheduler :
			public boost::fibers::algo::algorithm_with_properties< FiberProperties > {
		private:
			typedef std::deque< boost::fibers::context * >  rqueue_t;
			typedef boost::fibers::scheduler::ready_queue_t lqueue_t;

			static rqueue_t     	rqueue_;
			static std::mutex   	queueMutex;
			static rqueue_t     	rmqueue_;

			lqueue_t            	lqueue_{};
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

			virtual void awakened(boost::fibers::context * ctx, FiberProperties& props) noexcept {

				//dont push fiber when helper or the main fiber is passed in
				if (ctx->is_context(boost::fibers::type::pinned_context)) {
					lqueue_.push_back(*ctx);
				}
				else {

					int ctx_priority = props.GetPriority();
					bool mainRun = props.RunOnMain();

					ctx->detach();

					std::unique_lock< std::mutex > lk(queueMutex);

					//if it needs to run on the main thread
					if (mainRun) {
						rqueue_t::iterator i(std::find_if(rmqueue_.begin(), rmqueue_.end(),
							[ctx_priority, this](boost::fibers::context* c)
						{ return properties(c).GetPriority() < ctx_priority; }));

						rmqueue_.insert(i, ctx);
					}
					else {
						rqueue_t::iterator i(std::find_if(rqueue_.begin(), rqueue_.end(),
							[ctx_priority, this](boost::fibers::context* c)
						{ return properties(c).GetPriority() < ctx_priority; }));

						rqueue_.insert(i, ctx);
					}
				}
			}

			virtual boost::fibers::context * pick_next() noexcept {
				boost::fibers::context * ctx(nullptr);
				std::thread::id thisID = std::this_thread::get_id();

				std::unique_lock< std::mutex > lk(queueMutex);

				if (!rmqueue_.empty() && thisID == mainID) {
					ctx = rmqueue_.front();
					rmqueue_.pop_front();
					lk.unlock();
					BOOST_ASSERT(nullptr != ctx);
					boost::fibers::context::active()->attach(ctx);
				}
				else if (!rqueue_.empty()) {

					ctx = rqueue_.front();
					bool sr = this->properties(ctx).RunOnMain();
					int prior = this->properties(ctx).GetPriority();

					if (!sr) {

						rqueue_.pop_front();
						lk.unlock();
						BOOST_ASSERT(nullptr != ctx);
						boost::fibers::context::active()->attach(ctx);

					}
					else {

						rqueue_.pop_front();
						rqueue_t::iterator i(std::find_if(rmqueue_.begin(), rmqueue_.end(),
							[prior, this](boost::fibers::context* c)
						{ return properties(c).GetPriority() < prior; }));
						rmqueue_.insert(i, ctx);

						ctx = nullptr;

					}
				}
				if (ctx == nullptr) {

					lk.unlock();

					if (!lqueue_.empty()) {
						ctx = &lqueue_.front();
						lqueue_.pop_front();
					}

				}
				return ctx;
			}

			virtual bool has_ready_fibers() const noexcept {
				std::unique_lock< std::mutex > lock(queueMutex);
				return !rmqueue_.empty() || !rqueue_.empty() || !lqueue_.empty();
			}

			virtual void property_change(boost::fibers::context * ctx, FiberProperties & props) noexcept {
				if (!ctx->ready_is_linked()) {
					return;
				}

				// Found ctx: unlink it
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

		SoulScheduler::rqueue_t SoulScheduler::rqueue_{};
		SoulScheduler::rqueue_t SoulScheduler::rmqueue_{};
		std::mutex SoulScheduler::queueMutex{};


		//launches a thread that waits with a fiber conditional, meaning it still executes fibers despite waiting for a notify release
		void ThreadRun() {
			boost::fibers::use_scheduling_algorithm<SoulScheduler >();

			std::unique_lock<std::mutex> lock(Scheduler::detail::fiberMutex);
			threadCondition.wait(lock, []() { return 0 == Scheduler::detail::fiberCount && !detail::shouldRun; });
		}


	}

	void Terminate() {
		detail::fiberMutex.lock();
		detail::shouldRun = false;
		if (0 == --detail::fiberCount) {
			detail::fiberMutex.unlock();
			threadCondition.notify_all(); //notify all fibers waiting 
		}
		detail::fiberMutex.unlock();

		//yield this fiber until all the remaining work is done
		while (detail::fiberCount != 0) {
			boost::this_fiber::yield();
			threadCondition.notify_all();
		}

		//join all complete threads
		for (uint i = 0; i < threadCount; ++i) {
			threads[i].join();
		}

		delete[] threads;
	}

	void Init() {

		detail::mainID = std::this_thread::get_id();

		boost::fibers::use_scheduling_algorithm< detail::SoulScheduler >();

		//the main thread takes up one slot.
		threadCount = std::thread::hardware_concurrency() - 1;
		threads = new std::thread[threadCount];

		detail::fiberCount++;

		for (uint i = 0; i < threadCount; ++i) {
			threads[i] = std::thread(detail::ThreadRun);
		}
	}


	void Wait() {
		//could not be initialized if wait is called before an addTask
		detail::InitCheck();

		std::unique_lock<std::mutex> lock(*detail::holdMutex);
		detail::blockCondition->wait(lock, []() { return 0 == *detail::holdCount; });
	}

	void Defer() {
		detail::InitCheck();
		boost::this_fiber::yield();
	}

	bool Running() {
		return detail::shouldRun;
	}

};
