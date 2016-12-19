#pragma once

#include "Engine Core\BasicDependencies.h"
#include <boost/fiber/all.hpp>


//IMMEDIATE: Run the fiber immediatly with no context switch 
//Use Case: You will execute 100 tasks and will wait till they complete
//CONTINUE: Keep the current context and add the fiber to the queue 
//Use Case: You want to process other things while this function gets executed elsewhere
enum FiberPolicy { IMMEDIATE, CONTINUE };

namespace Scheduler {

	namespace detail {
		extern std::size_t fiberCount;
		extern std::mutex fiberMutex;

		//blocker for the holding count
		extern boost::fibers::fiber_specific_ptr<std::size_t> holdCount;
		extern boost::fibers::fiber_specific_ptr<std::mutex> holdMutex;
		extern boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable_any> blockCondition;
		void InitCheck();
	}

	void Init();
	void Terminate();


	template<typename Fn,
		typename ... Args>
		void AddTask(FiberPolicy policy, Fn && fn, Args && ... args) {


		detail::fiberMutex.lock();
		detail::fiberCount++;
		detail::fiberMutex.unlock();

		if (policy == IMMEDIATE) {

			detail::InitCheck();

			std::mutex* holdLock = detail::holdMutex.get();
			std::size_t* holdSize = detail::holdCount.get();
			boost::fibers::condition_variable_any* holdConditional = detail::blockCondition.get();

			holdLock->lock();
			(*holdSize)++;
			holdLock->unlock();

			boost::fibers::fiber(
				[&, holdLock, holdSize, holdConditional]() mutable {

				//prefix code

				///////////////////////////////////////////
				fn(std::forward<Args>(args)...);
				///////////////////////////////////////////

				//suffix code
				detail::fiberMutex.lock();
				detail::fiberCount--;
				detail::fiberMutex.unlock();

				holdLock->lock();
				(*holdSize)--;
				holdLock->unlock();
				holdConditional->notify_all();

			}).detach();
		}
		else {
			boost::fibers::fiber(
				[&]() mutable {

				//prefix code

				///////////////////////////////////////////
				fn(std::forward<Args>(args)...);
				///////////////////////////////////////////

				//suffix code
				detail::fiberMutex.lock();
				detail::fiberCount--;
				detail::fiberMutex.unlock();

			}).detach();
		}
	}

	//Blocks the fiber until all tasks with the IMMEDIATE policy have been executed
	void Wait();

};
