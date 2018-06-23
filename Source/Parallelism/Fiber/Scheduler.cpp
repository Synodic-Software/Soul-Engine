#include "Scheduler.h"
#include "SchedulerAlgorithm.h"

Scheduler::Scheduler(Property<uint>& threadCountIn) :
	shouldRun(true),
	fiberCount(0),
	threadCount(threadCountIn)
{

	mainID = std::this_thread::get_id();

	boost::fibers::use_scheduling_algorithm<SchedulerAlgorithm>(threadCount, true);

	//the main thread takes up one slot
	childThreads.resize(threadCount - 1);

	for (auto& thread : childThreads) {
		thread = std::thread([this] {
			ThreadRun();
		});
	}

	//init the main fiber specifics
	InitPointers();

	{
		std::scoped_lock<boost::fibers::mutex> incrementLock(fiberMutex);
		fiberCount++;
	}

	//suprisingly, the main fiber does not need to run on main.
	boost::this_fiber::properties<FiberProperties>().SetProperties(FiberPriority::UX, false); 
}

Scheduler::~Scheduler() {

	int polledCount;
	{
		std::scoped_lock<boost::fibers::mutex> lock(fiberMutex);
		shouldRun = false;
		--fiberCount;
		polledCount = fiberCount;
	}

	//all work is done, so notify all waiting threads
	if (polledCount == 0) {

		fiberCondition.notify_all();

	}
	//wait until all remaining work is done before spinlocking on join. Leave this thread available for processing
	else {

		std::unique_lock<boost::fibers::mutex> lock(fiberMutex);
		fiberCondition.wait(lock, [this]()
		{
			return 0 == fiberCount && !shouldRun;
		});

	}

	//join all complete threads
	for (auto& thread : childThreads) {
		thread.join();
	}

}

/* Initialize the fiber specific stuff. */
//TODO use fiber specific allocator
void Scheduler::InitPointers() {
	if (!blockMutex.get()) {
		blockMutex.reset(new boost::fibers::mutex);
	}
	if (!blockCount.get()) {
		blockCount.reset(new uint(0));
	}
	if (!blockCondition.get()) {
		blockCondition.reset(new boost::fibers::condition_variable);
	}
}

//block the current thread until all registered children complete
void Scheduler::Block() const {


	//get the current fibers stats for blocking
	uint* holdSize = blockCount.get();

	std::unique_lock<boost::fibers::mutex> lock(*blockMutex);
	blockCondition->wait(lock, [holdSize]()
	{
		return 0 == *holdSize;
	});

}

//Defers this current fiber.
void Scheduler::Defer() {

	boost::this_fiber::yield();

}

bool Scheduler::Running() const {

	return shouldRun;

}

/*
*    launches a thread that waits with a fiber conditional, meaning it still executes fibers
*    despite waiting for a notify release.
*/

void Scheduler::ThreadRun() {

	boost::fibers::use_scheduling_algorithm<SchedulerAlgorithm>(threadCount, false);

	boost::this_fiber::properties<FiberProperties>().SetProperties(FiberPriority::LOW, false);

	//continue processing until exit is called
	{
		std::unique_lock<boost::fibers::mutex> lock(fiberMutex);
		fiberCondition.wait(lock, [this]()
		{
			return 0 == fiberCount && !shouldRun;
		});
	}

}