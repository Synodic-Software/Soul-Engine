#include "Scheduler.h"
#include <boost/fiber/all.hpp>
#include <thread>
#include <condition_variable>

std::thread* threads;
unsigned int threadCount;

static std::size_t fiberCount{ 0 };
static bool shouldRun{ true };

static std::mutex fiberMutex{};
static boost::fibers::condition_variable_any condition{};

void ThreadRun() {
	boost::fibers::use_scheduling_algorithm< boost::fibers::algo::shared_work >();

	std::unique_lock<std::mutex> lock(fiberMutex);
	condition.wait(lock, []() { return 0 == fiberCount && !shouldRun; });
}

namespace Scheduler {

	void Terminate() {


		std::unique_lock<std::mutex> lock(fiberMutex);
		shouldRun = false;
		if (0 == --fiberCount) {
			lock.unlock();
			condition.notify_all(); //notify all fibers waiting 
		}

		while (fiberCount!=0) {
			boost::this_fiber::yield();
		}

		for (int i = 0; i < threadCount; ++i) {
			threads[i].join();
		}

		delete[] threads;

	}

	void Init() {

		boost::fibers::use_scheduling_algorithm< boost::fibers::algo::shared_work >();

		threadCount = std::thread::hardware_concurrency() - 1;  //the main thread takes up one slot.

		threads = new std::thread[threadCount];

		fiberCount++;

		for (int i = 0; i < threadCount; ++i) {
			threads[i] = std::thread(ThreadRun);
		}

	}

}
