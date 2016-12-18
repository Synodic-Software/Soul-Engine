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
		extern std::size_t holdCount;
		extern std::mutex holdMutex;
	}


	void Init();
	void Terminate();


	template<typename Fn,
		typename ... Args>
		void AddTask(FiberPolicy policy, Fn && fn, Args && ... args) {


		std::unique_lock<std::mutex> lock(detail::fiberMutex);
		detail::fiberCount++;
		lock.unlock();

		if (policy == IMMEDIATE) {

			std::unique_lock<std::mutex> hlock(detail::holdMutex);
			detail::holdCount++;
			hlock.unlock();

			boost::fibers::fiber(
				[&]() mutable {

				//prefix code


				///////////////////////////////////////////
				fn(std::forward<Args>(args)...);
				///////////////////////////////////////////

				//suffix code
				lock.lock();
				detail::fiberCount--;
				lock.unlock();

				hlock.lock();
				detail::fiberCount--;
				hlock.unlock();

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
				lock.lock();
				detail::fiberCount--;
				lock.unlock();

			}).detach();
		}
	}

	//Blocks the fiber untill all tasks with the IMMEDIATE policy have been executed
	void Wait();

};
