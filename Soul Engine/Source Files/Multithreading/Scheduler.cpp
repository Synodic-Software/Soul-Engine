#include "Scheduler.h"
#include <thread>

//Scheduler Variables//
static std::thread* threads;
static std::size_t threadCount{ 0 };

static bool shouldRun{ true };
static boost::fibers::condition_variable_any threadCondition{};



namespace Scheduler {

	namespace detail {
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

		class shared_priority :
			public boost::fibers::algo::algorithm_with_properties< priority_props > {
		private:
			typedef std::deque< boost::fibers::context * >  rqueue_t;
			typedef boost::fibers::scheduler::ready_queue_t lqueue_t;

			static rqueue_t     	rqueue_;
			static std::mutex   	rqueue_mtx_;
			static rqueue_t     	rmqueue_;

			lqueue_t            	lqueue_{};
			std::mutex              mtx_{};
			std::condition_variable cnd_{};
			bool                    flag_{ false };
			bool                    suspend_;

		public:
			shared_priority() = default;

			shared_priority(bool suspend) :
				suspend_{ suspend } {
			}

			shared_priority(shared_priority const&) = delete;
			shared_priority(shared_priority &&) = delete;

			shared_priority & operator=(shared_priority const&) = delete;
			shared_priority & operator=(shared_priority &&) = delete;

			virtual void awakened(boost::fibers::context * ctx, priority_props & props) noexcept {

				//dont push fiber when helper or the main fiber is passed in
				if (ctx->is_context(boost::fibers::type::pinned_context)) {
					lqueue_.push_back(*ctx);
				}
				else {
					ctx->detach();
					std::unique_lock< std::mutex > lk(rqueue_mtx_);

					int ctx_priority = props.get_priority();
					std::thread::id thisID = std::this_thread::get_id();

					//if it needs to run on the main thread
					if (props.get_main() && thisID == mainID) {
						rqueue_t::iterator i(std::find_if(rmqueue_.begin(), rmqueue_.end(),
							[ctx_priority, this](boost::fibers::context* c)
						{ return properties(c).get_priority() < ctx_priority; }));

						rmqueue_.insert(i, ctx);
					}
					else {
						rqueue_t::iterator i(std::find_if(rqueue_.begin(), rqueue_.end(),
							[ctx_priority, this](boost::fibers::context* c)
						{ return properties(c).get_priority() < ctx_priority; }));

						rqueue_.insert(i, ctx);
					}
				}
			}

			virtual boost::fibers::context * pick_next() noexcept {
				boost::fibers::context * ctx(nullptr);
				std::thread::id thisID = std::this_thread::get_id();

				std::unique_lock< std::mutex > lk(rqueue_mtx_);

				if (!rmqueue_.empty() && thisID == mainID) {
					ctx = rmqueue_.front();
					rmqueue_.pop_front();
					lk.unlock();
					BOOST_ASSERT(nullptr != ctx);
					boost::fibers::context::active()->attach(ctx);
				}
				else if (!rqueue_.empty()) {
					ctx = rqueue_.front();
					rqueue_.pop_front();
					lk.unlock();
					BOOST_ASSERT(nullptr != ctx);
					boost::fibers::context::active()->attach(ctx);
				}
				else {
					lk.unlock();
					if (!lqueue_.empty()) {
						ctx = &lqueue_.front();
						lqueue_.pop_front();
					}
				}
				return ctx;
			}

			virtual bool has_ready_fibers() const noexcept {
				std::unique_lock< std::mutex > lock(rqueue_mtx_);
				return !rmqueue_.empty() || !rqueue_.empty() || !lqueue_.empty();
			}

			virtual void property_change(boost::fibers::context * ctx, priority_props & props) noexcept {
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

		shared_priority::rqueue_t shared_priority::rqueue_{};
		shared_priority::rqueue_t shared_priority::rmqueue_{};
		std::mutex shared_priority::rqueue_mtx_{};


		//launches a thread that waits with a fiber conditional, meaning it still executes fibers despite waiting for a notify release
		void ThreadRun() {
			boost::fibers::use_scheduling_algorithm<shared_priority >();

			std::unique_lock<std::mutex> lock(Scheduler::detail::fiberMutex);
			threadCondition.wait(lock, []() { return 0 == Scheduler::detail::fiberCount && !shouldRun; });
		}


	}

	void Terminate() {
		detail::fiberMutex.lock();
		shouldRun = false;
		if (0 == --detail::fiberCount) {
			detail::fiberMutex.unlock();
			threadCondition.notify_all(); //notify all fibers waiting 
		}

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

		boost::fibers::use_scheduling_algorithm< detail::shared_priority >();

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
		boost::this_fiber::yield();
	}

};
