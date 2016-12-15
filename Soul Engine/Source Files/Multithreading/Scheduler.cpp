#include "Scheduler.h"
#include <boost/fiber/all.hpp>
#include <thread>

std::thread* threads;
unsigned int threadCount;

static std::size_t fiberCount{ 0 };
static bool shouldRun{ true };

static std::mutex fiberMutex{};
static boost::fibers::condition_variable_any condition{};
typedef std::unique_lock< std::mutex > lock_t;

void ThreadRun() {
	boost::fibers::use_scheduling_algorithm< boost::fibers::algo::shared_work >();


	lock_t lk(fiberMutex);
	condition.wait(lk, []() { return 0 == fiberCount && !shouldRun; });
}

namespace Scheduler {

	void Terminate() {
		

		lock_t lk(fiberMutex);
		shouldRun = false;
		if (0 == --fiberCount) { 
			lk.unlock();
			condition.notify_all(); //notify all fibers waiting 
		}

		condition.wait(lk, []() { return 0 == fiberCount && !shouldRun; });

		for (int i = 0; i < threadCount; ++i) {
			threads[i].join();
		}

		delete threads;

	}

	void Init() {

		boost::fibers::use_scheduling_algorithm< boost::fibers::algo::shared_work >();

		threadCount = std::thread::hardware_concurrency()-1;  //the main thread takes up one slot.

		threads = new std::thread[threadCount];

		fiberCount++;

		for (int i = 0; i < threadCount; ++i) {
			threads[i] = std::thread(ThreadRun);
		}

	}

}
