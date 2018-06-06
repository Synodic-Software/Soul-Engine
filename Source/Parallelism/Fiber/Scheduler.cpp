#include "Scheduler.h"
#include "SchedulerAlgorithm.h"


/*
*    clean up the block datatype (needs 64 alignment)
*    @param [in,out]	ptr	If non-null, the pointer.
*/

void CleanUpAlignedCondition(boost::fibers::condition_variable* ptr) {
	ptr->~condition_variable();

	//TODO: Make this aligned_alloc with c++17, not visual studio specific code
	_aligned_free(ptr);
}



/* Initializes this object. */
Scheduler::Scheduler(Property<int>& threadCount) :
	shouldRun(true),
	fiberCount(0),
	blockCondition(CleanUpAlignedCondition)
{

	mainID = std::this_thread::get_id();

	boost::fibers::use_scheduling_algorithm< SchedulerAlgorithm >();

	//the main thread takes up one slot
	threads.resize(threadCount - 1);

	for (auto& thread : threads) {
		thread = std::thread([this] {
			ThreadRun();
		});
	}

	//init the main fiber specifics
	fiberCount++;
	InitPointers();

}

Scheduler::~Scheduler() {

	fiberMutex.lock();
	shouldRun = false;
	--fiberCount;
	fiberMutex.unlock();

	//wait until all remaining work is done
	std::unique_lock<boost::fibers::mutex> lock(fiberMutex);
	fiberCondition.wait(lock, [this]()
	{
		return 0 == fiberCount && !shouldRun;
	});

	//join all complete threads
	for (auto& thread : threads) {
		thread.join();
	}

}

/* Initialize the fiber specific stuff. */
void Scheduler::InitPointers() {
	if (!blockMutex.get()) {
		blockMutex.reset(new boost::fibers::mutex);
	}
	if (!blockCount.get()) {
		blockCount.reset(new std::size_t(0));
	}
	if (!blockCondition.get()) {

		//TODO: Make this aligned_alloc with c++17, not visual studio specific code
		boost::fibers::condition_variable* newData =
			static_cast<boost::fibers::condition_variable*>(_aligned_malloc(sizeof(boost::fibers::condition_variable), 64)); //needs 64 alignment
		new (newData) boost::fibers::condition_variable();
		blockCondition.reset(newData);
	}
}

/* Blocks this object. */
void Scheduler::Block() {


	//get the current fibers stats for blocking
	std::size_t* holdSize = blockCount.get();

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

/*
 *    Runnings this object.
 *    @return	True if it succeeds, false if it fails.
 */

bool Scheduler::Running() const {

	return shouldRun;

}

/*
*    launches a thread that waits with a fiber conditional, meaning it still executes fibers
*    despite waiting for a notify release.
*/

void Scheduler::ThreadRun() {
	boost::fibers::use_scheduling_algorithm<SchedulerAlgorithm >();

	std::unique_lock<boost::fibers::mutex> lock(fiberMutex);
	fiberCondition.wait(lock, [this]()
	{
		return 0 == fiberCount && !shouldRun;
	});
}