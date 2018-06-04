#include "Scheduler.h"
#include "SchedulerAlgorithm.h"

#include "Composition/Event/EventManager.h"
#include <cassert>


/*
*    clean up the block datatype (needs 64 alignment)
*    @param [in,out]	ptr	If non-null, the pointer.
*/

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
	fiberCount(0),
	shouldRun(true),
	blockCondition(CleanUpAlignedCondition)
{

	mainID = std::this_thread::get_id();

	boost::fibers::use_scheduling_algorithm< SchedulerAlgorithm >();

	//the main thread takes up one slot
	threads.resize(std::thread::hardware_concurrency() - 1);

	fiberCount++;

	for(auto& thread : threads) {
		thread = std::thread(
			[this] { ThreadRun(); }
		);
	}

	//init the main fiber specifics
	InitPointers();

}

Scheduler::~Scheduler() {

	fiberMutex.lock();
	shouldRun = false;
	--fiberCount;
	fiberMutex.unlock();

	//yield this fiber until all the remaining work is done

	bool run = true;
	while (run) {
		fiberMutex.lock();
		if (fiberCount == 0) {
			run = false;
			fiberMutex.unlock();
			threadCondition.notify_all();
		}
		else {
			fiberMutex.unlock();
			boost::this_fiber::yield();
		}
	}

	//join all complete threads
	for (uint i = 0; i < threads.size(); ++i) {
		threads[i].join();
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
	blockCondition->wait(lock, [=]() { return 0 == *holdSize; });

	assert(*holdSize == 0);


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

	EventManager::Emit("Thread", "Initialize");

	std::unique_lock<std::mutex> lock(fiberMutex);
	threadCondition.wait(lock, [this]()
	{
		return 0 == fiberCount && !shouldRun;
	});
}