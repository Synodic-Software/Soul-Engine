#include "Scheduler.h"
#include <thread>


static std::thread* threads;
static std::size_t threadCount{ 0 };

static bool shouldRun{ true };

static boost::fibers::condition_variable_any threadCondition{};

//launches a thread that waits with a fiber conditional, meaning it still executes fibers despite waiting for a notify release
void ThreadRun() {
	boost::fibers::use_scheduling_algorithm< boost::fibers::algo::shared_work >();

	std::unique_lock<std::mutex> lock(Scheduler::detail::fiberMutex);
	threadCondition.wait(lock, []() { return 0 == Scheduler::detail::fiberCount && !shouldRun; });
}

namespace Scheduler {

	namespace detail {
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
		boost::fibers::use_scheduling_algorithm< boost::fibers::algo::shared_work >();

		//the main thread takes up one slot.
		threadCount = std::thread::hardware_concurrency() - 1;
		threads = new std::thread[threadCount];

		detail::fiberCount++;

		for (uint i = 0; i < threadCount; ++i) {
			threads[i] = std::thread(ThreadRun);
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
