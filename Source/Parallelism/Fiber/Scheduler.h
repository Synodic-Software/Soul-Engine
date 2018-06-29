#pragma once

#include <boost/fiber/fss.hpp>
#include <boost/fiber/condition_variable.hpp>

#include <thread>
#include <mutex>

#include "Core/Utility/Property/Property.h"
#include "FiberParameters.h"
#include "FiberProperties.h"
#include "Core/Utility/Types.h"

#undef CreateWindow

//TODO: Implement boost Fiber: Work_stealing


class Scheduler {

public:

	Scheduler(Property<uint>&);
	~Scheduler();

	Scheduler(Scheduler const&) = delete;
	void operator=(Scheduler const&) = delete;


	template<typename Fn, typename ... Args>
	void AddTask(FiberParameters, Fn &&, Args && ...);

	void Block() const;
	static void Defer();
	bool Running() const;


private:

	void InitPointers();
	void ThreadRun();

	std::thread::id mainID;

	bool shouldRun;

	uint fiberCount;
	boost::fibers::mutex fiberMutex;
	boost::fibers::condition_variable fiberCondition; //no spurious wakeups with the boost::fibers version

	boost::fibers::fiber_specific_ptr<uint> blockCount;
	boost::fibers::fiber_specific_ptr<boost::fibers::mutex> blockMutex;
	boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable> blockCondition;  //no spurious wakeups with the boost::fibers version

	Property<uint>& threadCount;
	std::vector<std::thread> childThreads;

};

/*
 * Adds a task.
 *
 * @tparam	Fn  	Type of the function.
 * @tparam	Args	Type of the arguments.
 * @param 		  	params	Options for controlling the operation.
 * @param [in,out]	fn	  	The function.
 * @param 		  	args  	Function Arguments The arguments.
 */

template<typename Fn, typename ... Args>
void Scheduler::AddTask(FiberParameters params, Fn && fn, Args && ... args) {

	//increment the global fiber count
	{
		std::scoped_lock<boost::fibers::mutex> incrementLock(fiberMutex);
		fiberCount++;
	}

	//create the launch type
	auto launchType = params.swap ? boost::fibers::launch::dispatch : boost::fibers::launch::post;

	auto executeFiber = [this, fn, params]() mutable {

		//TODO once fiber properties are changes/you can enque fibers to a specific scheduler, remove the yield
		boost::this_fiber::properties<FiberProperties>().SetProperties(params.priority, params.needsMainThread);
		boost::this_fiber::yield();

		//initialze the data associated with the fiber
		InitPointers();

		/////////////////////////////////////////////
		std::invoke(fn, std::forward<Args>(args)...);
		/////////////////////////////////////////////

		//All children fibers must be completed
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

	if (params.attach) {

		//grab the parent fiber data and the relevant locks
		boost::fibers::mutex* holdLock = blockMutex.get();
		uint* holdSize = blockCount.get();
		boost::fibers::condition_variable* holdConditional = blockCondition.get();

		//increment the parent fiber count
		{
			std::scoped_lock<boost::fibers::mutex> incrementLock(*holdLock);
			(*holdSize)++;
		}

		boost::fibers::fiber fiber(
			launchType,
			[executeFiber, holdLock, holdSize, holdConditional]() mutable {

			executeFiber();

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
	else{

		boost::fibers::fiber fiber(
			launchType,
			executeFiber
		);

		fiber.detach();

	}

}