#include "Scheduler.h"
#include <thread>


static std::thread* threads;
static std::size_t threadCount{ 0 };

static bool shouldRun{ true };

static std::mutex fiberMutex{};
static boost::fibers::condition_variable_any condition{};

void ThreadRun() {
	boost::fibers::use_scheduling_algorithm< boost::fibers::algo::shared_work >();

	std::unique_lock<std::mutex> lock(fiberMutex);
	condition.wait(lock, []() { return 0 == Scheduler::detail::fiberCount && !shouldRun; });
}

namespace Scheduler {

	namespace detail {
		std::size_t fiberCount =0;


	}

	void Terminate() {
		std::unique_lock<std::mutex> lock(fiberMutex);
		shouldRun = false;
		if (0 == --detail::fiberCount) {
			lock.unlock();
			condition.notify_all(); //notify all fibers waiting 
		}

		while (detail::fiberCount!=0) {
			boost::this_fiber::yield();
		}
		 
		for (uint i = 0; i < threadCount; ++i) {
			threads[i].join();
		}

		delete[] threads;
	}

	void Init() {
		boost::fibers::use_scheduling_algorithm< boost::fibers::algo::shared_work >();

		threadCount = std::thread::hardware_concurrency() - 1;  //the main thread takes up one slot.
		threads = new std::thread[threadCount];

		detail::fiberCount++;

		for (uint i = 0; i < threadCount; ++i) {
			threads[i] = std::thread(ThreadRun);
		}
	}

};
