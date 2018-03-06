#include "Scheduler.h"
#include "SchedulerAlgorithm.h"

#include "Events/EventManager.h"



/*
*    clean up the block datatype (needs 64 alignment)
*    @param [in,out]	ptr	If non-null, the pointer.
*/

void CleanUpMutex(std::mutex* ptr) {
	ptr->~mutex();
	delete ptr;
}

/*
*    clean up the block datatype (needs 64 alignment)
*    @param [in,out]	ptr	If non-null, the pointer.
*/

void CleanUpAlignedCondition(boost::fibers::condition_variable_any* ptr) {
	ptr->~condition_variable_any();

	//TODO: Make this aligned_alloc with c++17, not visual studio specific code
	_aligned_free(ptr);
}



/* Initializes this object. */
Scheduler::Scheduler() :
	fiberCount(0),
	shouldRun(true),
	holdMutex(CleanUpMutex),
	blockCondition(CleanUpAlignedCondition)
{

	mainID = std::this_thread::get_id();

	boost::fibers::use_scheduling_algorithm< SchedulerAlgorithm >();

	//the main thread takes up one slot
	threads.resize(std::thread::hardware_concurrency() - 1);

	fiberCount++;

	for (uint i = 0; i < threads.size(); ++i) {
		threads[i] = std::thread(
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
	if (!holdMutex.get()) {
		holdMutex.reset(new std::mutex);
	}
	if (!holdCount.get()) {
		holdCount.reset(new std::size_t(0));
	}
	if (!blockCondition.get()) {

		//TODO: Make this aligned_alloc with c++17, not visual studio specific code
		boost::fibers::condition_variable_any* newData =
			static_cast<boost::fibers::condition_variable_any*>(_aligned_malloc(sizeof(boost::fibers::condition_variable_any), 64)); //needs 64 alignment
		new (newData) boost::fibers::condition_variable_any();
		blockCondition.reset(newData);
	}
}

/* Blocks this object. */
void Scheduler::Block() {


	//get the current fibers stats for blocking
	std::size_t* holdSize = holdCount.get();

	std::unique_lock<std::mutex> lock(*holdMutex);
	blockCondition->wait(lock, [=]() { return 0 == *holdSize; });

	assert(*holdSize == 0);


}

/* Defers this object. */
void Scheduler::Defer() {

#ifndef	SOUL_SINGLE_STACK

	boost::this_fiber::yield();

#endif

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