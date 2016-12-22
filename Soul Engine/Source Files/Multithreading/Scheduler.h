#pragma once

#include "Engine Core\BasicDependencies.h"
#include <boost/fiber/all.hpp>
#include <boost/fiber/algo/algorithm.hpp>

/*
IMMEDIATE: Run the fiber immediatly with no context switch
Use Case: You will execute 100 tasks and will wait till they complete
CONTINUE: Keep the current context and add the fiber to the queue
Use Case: You want to process other things while this function gets executed elsewhere
*/
enum FiberPolicy { LAUNCH_IMMEDIATE, LAUNCH_CONTINUE };

/*
FIBER_HIGH: A high priority task.
FIBER_LOW: A comparitivley low priority task.
*/
enum FiberPriority { FIBER_HIGH, FIBER_LOW };

namespace Scheduler {

	namespace detail {

		extern std::size_t fiberCount;
		extern std::mutex fiberMutex;

		//blocker for the holding count
		extern boost::fibers::fiber_specific_ptr<std::size_t> holdCount;
		extern boost::fibers::fiber_specific_ptr<std::mutex> holdMutex;
		extern boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable_any> blockCondition;

		//initializes all fiber_specific_ptrs if they havnt been initialized
		void InitCheck();

		//property class for the custom scheduler
		class priority_props : public boost::fibers::fiber_properties {
		public:
			priority_props(boost::fibers::context * context) :
				fiber_properties(context),
				priority(0),
				runMain(false){
			}

			int get_priority() const {
				return priority;
			}

			bool get_main() const {
				return runMain;
			}

			//setting the priority needs a notify update
			void set_priority(int p,bool m) {
				if (p != priority||m!= runMain) {
					priority = p;
					runMain = m;
					notify();
				}
			}

		private:
			int priority;
			int runMain;
		};

	}

	//Initialize the multithreaded scheduler
	void Init();

	//Terminate the multithreaded scheduler
	void Terminate();

	/*
	Add a task to the fiber system to be executed concurrently

	policy: The fiber policy after running the segment
	priority: Fiber execution priority
	runsOnMain: Requirement that this function runs on the main thread
	Fn && fn, Args && ... args:  The lambda to be executed.
	*/
	template<typename Fn,
		typename ... Args>
		void AddTask(FiberPolicy policy, FiberPriority priority, bool runsOnMain, Fn && fn, Args && ... args) {

		//this thread increments the locks, the launched fiber implements the decrement
		detail::fiberMutex.lock();
		detail::fiberCount++;
		detail::fiberMutex.unlock();

		//only difference is the hold lock increment
		if (policy == LAUNCH_IMMEDIATE) {

			detail::InitCheck();

			std::mutex* holdLock = detail::holdMutex.get();
			std::size_t* holdSize = detail::holdCount.get();
			boost::fibers::condition_variable_any* holdConditional = detail::blockCondition.get();

			holdLock->lock();
			(*holdSize)++;
			holdLock->unlock();

			//lambda wrapping the called function with other information
			boost::fibers::fiber fiber(
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

			});

			detail::priority_props & props(fiber.properties< detail::priority_props >());
			props.set_priority(priority, runsOnMain);
			fiber.detach();
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

	//Blocks the fiber until all tasks with the LAUNCH_IMMEDIATE policy have been executed
	void Wait();

	//Yields the current fiber to the scheduler
	void Defer();

};
