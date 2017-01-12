#pragma once

//removes all fiber specifics and calls functions immediatly by pointer 
//(all tasks that repeat and yield will need to be removed)
//#define SOUL_SINGLE_STACK

#include <boost/fiber/all.hpp>

#include <iostream>

//TODO: Implement boost 1.63 Fiber: Work_stealing

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

	//User: Do Not Touch
	namespace detail {

		extern std::thread::id mainID;

		extern std::size_t fiberCount;
		extern std::mutex fiberMutex;

		//blocker for the holding count
		extern boost::fibers::fiber_specific_ptr<std::size_t>* holdCount;
		extern boost::fibers::fiber_specific_ptr<std::mutex>* holdMutex;
		extern boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable_any>* blockCondition;

		//initializes all fiber_specific_ptrs if they havnt been initialized
		void InitPointers();

		//property class for the custom scheduler
		class FiberProperties : public boost::fibers::fiber_properties {
		public:
			FiberProperties(boost::fibers::context * context) :
				fiber_properties(context),
				priority(0),
				runOnMain(false) {
			}

			//read priority
			int GetPriority() const {
				return priority;
			}

			//read shouldRunOnMain
			bool RunOnMain() const {
				return runOnMain;
			}

			//setting the priority needs a notify update
			void SetPriority(int p, bool m) {
				if (p != priority || m != runOnMain) {
					priority = p;
					runOnMain = m;
					notify();
				}
			}

		private:
			int priority;
			bool runOnMain;
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

#ifndef	SOUL_SINGLE_STACK

			std::mutex* holdLock = detail::holdMutex->get();
			std::size_t* holdSize = detail::holdCount->get();
			boost::fibers::condition_variable_any* holdConditional = detail::blockCondition->get();

			holdLock->lock();
			(*holdSize)++;
			holdLock->unlock();


			//lambda wrapping the called function with other information
			boost::fibers::fiber fiber(
				[&, holdLock, holdSize, holdConditional]() mutable {

				//prefix code
				detail::InitPointers();

#endif
				//assert that the function is executing on the right thread
				assert(!fiber.properties< detail::FiberProperties >().RunOnMain() ||
					(fiber.properties< detail::FiberProperties >().RunOnMain() && detail::mainID == std::this_thread::get_id()));

				///////////////////////////////////////////
				fn(std::forward<Args>(args)...);
				///////////////////////////////////////////

				//suffix code
				detail::fiberMutex.lock();
				detail::fiberCount--;
				detail::fiberMutex.unlock();

#ifndef	SOUL_SINGLE_STACK

				holdLock->lock();
				(*holdSize)--;
				holdLock->unlock();
				holdConditional->notify_all();

			});

			detail::FiberProperties& props(fiber.properties< detail::FiberProperties >());
			props.SetPriority(priority, runsOnMain);
			fiber.detach();

#endif

		}
		else {

#ifndef	SOUL_SINGLE_STACK

			boost::fibers::fiber fiber(
				[&]() mutable {

				//prefix code
				detail::InitPointers();

#endif
				//assert that the function is executing on the right thread
				assert(!fiber.properties< detail::FiberProperties >().RunOnMain() ||
					(fiber.properties< detail::FiberProperties >().RunOnMain() && detail::mainID == std::this_thread::get_id()));

				///////////////////////////////////////////
				fn(std::forward<Args>(args)...);
				///////////////////////////////////////////

				//suffix code
				detail::fiberMutex.lock();
				detail::fiberCount--;
				detail::fiberMutex.unlock();

#ifndef	SOUL_SINGLE_STACK

			});

			detail::FiberProperties& props(fiber.properties< detail::FiberProperties >());
			props.SetPriority(priority, runsOnMain);
			fiber.detach();

#endif

		}
	}

	//Blocks the fiber until all tasks with the LAUNCH_IMMEDIATE policy have been executed
	void Block();

	//Yields the current fiber to the scheduler
	void Defer();

	//Returns the running state of the scheduler. Useful for functions that want to run the lifespan of the engine
	bool Running();
};
