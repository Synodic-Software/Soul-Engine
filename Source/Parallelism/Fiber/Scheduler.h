#pragma once

#include <boost/fiber/fss.hpp>
#include <boost/fiber/condition_variable.hpp>

#include <thread>
#include <mutex>

#include "Core/Utility/Property/Property.h"

//TODO: Implement boost Fiber: Work_stealing

/*
IMMEDIATE: Schedule the fiber with the garuntee that you will
Use Case: You will execute a task and wait upon its completion later.
CONTINUE: Keep the current context and add the fiber to the queue
Use Case: You want to process other things while this function gets executed elsewhere
*/
enum FiberPolicy { LAUNCH_IMMEDIATE, LAUNCH_CONTINUE };

/*
FIBER_HIGH: A high priority task.
FIBER_LOW: A comparitivley low priority task.
*/
enum FiberPriority { FIBER_HIGH, FIBER_LOW, UX };

class Scheduler {

public:

	Scheduler(Property<int>&);
	~Scheduler();

	Scheduler(Scheduler const&) = delete;
	void operator=(Scheduler const&) = delete;


	template<typename Fn, typename ... Args>
	void AddTask(FiberPolicy, FiberPriority, bool, Fn &&, Args && ...);

	void Block();
	void Defer();
	bool Running() const;


private:


	void InitPointers();
	void ThreadRun();

	std::thread::id mainID;

	bool shouldRun;
	
	std::size_t fiberCount;
	boost::fibers::mutex fiberMutex;
	boost::fibers::condition_variable fiberCondition;

	boost::fibers::fiber_specific_ptr<std::size_t> blockCount;
	boost::fibers::fiber_specific_ptr<boost::fibers::mutex> blockMutex;
	boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable> blockCondition;

	std::vector<std::thread> threads;

};


/*
*    Adds a task.
*    @param 		 	policy	  	The fiber policy for running the segment
*    @param 		 	priority  	Fiber execution priority
*    @param 		 	runsOnMain	Requirement that this function runs on the main thread
*    @param [in,out]	fn		  	The function.
*    @param 		 	args	  	Function Arguments The arguments.
*	 @
*/

template<typename Fn, typename ... Args>
void Scheduler::AddTask(const FiberPolicy policy, const FiberPriority priority, const bool runsOnMain, Fn && fn, Args && ... args) {

	//increment the global fiber count
	{
		std::scoped_lock<boost::fibers::mutex> incrementLock(fiberMutex);
		fiberCount++;
	}

	auto fiberExecute = [this, fn]() mutable {

		//initialze the data associated with the fiber
		InitPointers();

		/////////////////////////////////////////////
		std::invoke(fn, std::forward<Args>(args)...);
		/////////////////////////////////////////////

		//user may have forgoten a block call
		Block();

		//execution finished, decrease the global fiber count
		size_t remainingFibers;
		{
			std::scoped_lock<boost::fibers::mutex> incrementLock(fiberMutex);
			fiberCount--;
			remainingFibers = fiberCount;
		}

		if (remainingFibers == 0) {
			fiberCondition.notify_all();
		}

	};

	//only difference is the hold lock increment
	if (policy == LAUNCH_IMMEDIATE) {

		//grab the parent fiber data and the relevant locks
		boost::fibers::mutex* holdLock = blockMutex.get();
		std::size_t* holdSize = blockCount.get();
		boost::fibers::condition_variable* holdConditional = blockCondition.get();

		//increment the parent fiber count
		{
			std::scoped_lock<boost::fibers::mutex> incrementLock(*holdLock);
			(*holdSize)++;
		}

		boost::fibers::fiber fiber(
			[fiberExecute, holdLock, holdSize, holdConditional]() mutable {

			fiberExecute();

			//decrement the parent fiber count and notify the parent if it reached 0
			size_t remainingHolds;
			{
				std::scoped_lock<boost::fibers::mutex> incrementLock(*holdLock);
				(*holdSize)--;
				remainingHolds = *holdSize;
			}

			if (remainingHolds == 0) {
				holdConditional->notify_all();
			}

		});
		fiber.detach();

	}
	else {

		boost::fibers::fiber fiber(
			fiberExecute
		);
		fiber.detach();

	}
}