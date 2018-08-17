#pragma once

#include <boost/fiber/fss.hpp>
#include <boost/fiber/condition_variable.hpp>

#include <vector>
#include <thread>
#include <mutex>

#include "Core/Utility/Property/Property.h"
#include "FiberParameters.h"
#include "FiberProperties.h"
#include "Core/Utility/Types.h"

//TODO: unlink the windows headers from the project/cmake https://github.com/Synodic-Software/Soul-Engine/issues/62
#undef CreateWindow
#undef Yield


class Scheduler {

public:

	Scheduler(Property<uint>&);
	~Scheduler();

	Scheduler(Scheduler const&) = delete;
	void operator=(Scheduler const&) = delete;

	template<typename Fn, typename ... Args>
	void AddTask(FiberParameters, Fn &&, Args && ...);

	template<typename Fn, typename ... Args>
	void ForEachThread(FiberPriority, Fn &&, Args && ...);

	void Block() const;
	static void Yield();

private:

	template< typename Fn >
	static void LaunchFiber(FiberParameters&, Fn &&);

	void InitPointers();
	void ThreadRun();

	bool shouldRun_; //flag to help coordniate destruction

	uint fiberCount_;
	boost::fibers::mutex fiberMutex_;
	boost::fibers::condition_variable fiberCondition_; //no spurious wakeups with the boost::fibers version

	boost::fibers::fiber_specific_ptr<uint> blockCount_;
	boost::fibers::fiber_specific_ptr<boost::fibers::mutex> blockMutex_;
	boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable> blockCondition_;  //no spurious wakeups with the boost::fibers version

	Property<uint>& threadCount_;
	std::vector<std::thread> childThreads_;

};

/*
 * Adds a task to the scheduler to be executed on any available thread.
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
		std::scoped_lock<boost::fibers::mutex> incrementLock(fiberMutex_);
		fiberCount_++;
	}

	auto executeFiber = [this, fn, params]() mutable {

		//init the data associated with the fiber
		InitPointers();

		/////////////////////////////////////////////
		std::invoke(fn, std::forward<Args>(args)...);
		/////////////////////////////////////////////

		//All children fibers must be completed
		Block();

		//execution finished, decrease the global fiber count
		size_t remainingFibers;
		{
			std::scoped_lock<boost::fibers::mutex> incrementLock(fiberMutex_);
			fiberCount_--;
			remainingFibers = fiberCount_;
		}

		if (remainingFibers == 0) {
			fiberCondition_.notify_all();
		}

	};

	if (params.shouldBlock_) {

		//grab the parent fiber data and the relevant locks
		boost::fibers::mutex* holdLock = blockMutex_.get();
		uint* holdSize = blockCount_.get();
		boost::fibers::condition_variable* holdConditional = blockCondition_.get();

		//increment the parent fiber count
		{
			std::scoped_lock<boost::fibers::mutex> incrementLock(*holdLock);
			(*holdSize)++;
		}

		LaunchFiber(
			params, 
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

	}
	else {

		LaunchFiber(
			params, 
			executeFiber
		);

	}

}

template<typename Fn, typename ... Args>
void Scheduler::ForEachThread(FiberPriority priority, Fn && fn, Args && ... args) {

	//immediately enter to provide block scope to the perthread tasks
	FiberParameters subParams(false);	
	AddTask(subParams, [this, priority, fn]() {

		FiberParameters usedParams(true,priority);

		for (uint threadIndex = 0; threadIndex < threadCount_; ++threadIndex) {

			usedParams.requiredThread_ = threadIndex;
			AddTask(usedParams, fn, std::forward<Args>(args)...);

		}

	});

}

template< typename Fn >
void Scheduler::LaunchFiber(FiberParameters& params, Fn && func) {

	if (params.post_) {

		boost::fibers::fiber fiber(boost::fibers::launch::post, func);

		fiber.properties<FiberProperties>().SetProperties(params.priority_, params.requiredThread_); //fiber not pushed to shared queues until properties are set
		fiber.detach();

	}
	else {

		//since the fiber is not awoken like a posted fiber, and immediatly entered, its property should not be new
		boost::fibers::fiber fiber(boost::fibers::launch::dispatch, func);

		//Don't let fiber fall out of scope (error). If already executed with swap, destruct the context
		fiber.join();

	}

}