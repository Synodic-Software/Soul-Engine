#include "Scheduler.h"
#include <thread>


static std::thread* threads;
static std::size_t threadCount{ 0 };

static bool shouldRun{ true };

static boost::fibers::condition_variable_any threadCondition{};
static boost::fibers::condition_variable_any blockCondition{};


void ThreadRun() {
	boost::fibers::use_scheduling_algorithm< boost::fibers::algo::shared_work >();

	std::unique_lock<std::mutex> lock(Scheduler::detail::fiberMutex);
	threadCondition.wait(lock, []() { return 0 == Scheduler::detail::fiberCount && !shouldRun; });
}

namespace Scheduler {

	namespace detail {
		std::size_t fiberCount =0;
		std::mutex fiberMutex;
		std::size_t holdCount=0;
		std::mutex holdMutex;
	}

	void Terminate() {
		std::unique_lock<std::mutex> lock(detail::fiberMutex);
		shouldRun = false;
		if (0 == --detail::fiberCount) {
			lock.unlock();
			threadCondition.notify_all(); //notify all fibers waiting 
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


	void Wait() {
		std::unique_lock<std::mutex> lock(detail::holdMutex);
		blockCondition.wait(lock, []() { return 0 == detail::holdCount; });
	}

};
