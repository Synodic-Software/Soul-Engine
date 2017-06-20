//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Multithreading\Scheduler.h.
//Declares the scheduler class.

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
//Values that represent fiber policies.
enum FiberPolicy { LAUNCH_IMMEDIATE, LAUNCH_CONTINUE };

/*
FIBER_HIGH: A high priority task.
FIBER_LOW: A comparitivley low priority task.
*/
//Values that represent fiber priorities.
enum FiberPriority { FIBER_HIGH, FIBER_LOW };

//.
namespace Scheduler {

	//User: Do Not Touch
	namespace detail {

		//Identifier for the main
		extern std::thread::id mainID;

		//Number of fibers
		extern std::size_t fiberCount;
		//The fiber mutex
		extern std::mutex fiberMutex;

		//blocker for the holding count.
		extern boost::fibers::fiber_specific_ptr<std::size_t>* holdCount;
		//The hold mutex
		extern boost::fibers::fiber_specific_ptr<std::mutex>* holdMutex;
		//The block condition
		extern boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable_any>* blockCondition;

		//initializes all fiber_specific_ptrs if they havnt been initialized.
		void InitPointers();

		//property class for the custom scheduler.
		class FiberProperties : public boost::fibers::fiber_properties {
		public:
			FiberProperties(boost::fibers::context * context) :
				fiber_properties(context),
				priority(0),

				//---------------------------------------------------------------------------------------------------
				//Constructor.
				//@param	parameter1	The first parameter.

				runOnMain(false) {
			}

			//---------------------------------------------------------------------------------------------------
			//read priority.
			//@return	The priority.

			int GetPriority() const {
				return priority;
			}

			//---------------------------------------------------------------------------------------------------
			//read shouldRunOnMain.
			//@return	True if it succeeds, false if it fails.

			bool RunOnMain() const {
				return runOnMain;
			}

			//---------------------------------------------------------------------------------------------------
			//setting the priority needs a notify update.
			//@param	p	An int to process.
			//@param	m	True to m.

			void SetPriority(int p, bool m) {
				if (p != priority || m != runOnMain) {
					priority = p;
					runOnMain = m;
					notify();
				}
			}

		private:
			//The priority
			int priority;
			//True to run on main
			bool runOnMain;
		};

	}

	//Initialize the multithreaded scheduler.
	void Initialize();

	//Terminate the multithreaded scheduler.
	void Terminate();

	//Takes in a member function and the object it is affiliated with
	template <typename T,
		typename ... Args>

		//---------------------------------------------------------------------------------------------------
		//Adds a task.
		//@param 		 	policy	  	The policy.
		//@param 		 	priority  	The priority.
		//@param 		 	runsOnMain	True to runs on main.
		//@param [in,out]	instance  	If non-null, the instance.
		//@param [in,out]	func	  	If non-null, the function.

		void AddTask(FiberPolicy policy, FiberPriority priority, bool runsOnMain, T *instance, void(T::*func)(Args...) const) {
		AddTask(policy, priority, runsOnMain, [=](Args... args) {
			(instance->*func)(args...);
		});
	}
	
	//Takes in a member function and the object it is affiliated with
	template <typename T,
		typename ... Args>

		//---------------------------------------------------------------------------------------------------
		//Adds a task.
		//@param 		 	policy	  	The policy.
		//@param 		 	priority  	The priority.
		//@param 		 	runsOnMain	True to runs on main.
		//@param [in,out]	instance  	If non-null, the instance.
		//@param [in,out]	func	  	If non-null, the function.

		void AddTask(FiberPolicy policy, FiberPriority priority, bool runsOnMain, T *instance, void(T::*func)(Args...)) {
		AddTask(policy, priority, runsOnMain, [=](Args... args) {
			(instance->*func)(args...);
		});
	}


	/*
	Add a task to the fiber system to be executed concurrently

	policy: The fiber policy after running the segment
	priority: Fiber execution priority
	runsOnMain: Requirement that this function runs on the main thread
	Fn && fn, Args && ... args:  The lambda to be executed.
	*/
	template<typename Fn,
		typename ... Args>

		//---------------------------------------------------------------------------------------------------
		//Adds a task.
		//@param 		 	policy	  	The policy.
		//@param 		 	priority  	The priority.
		//@param 		 	runsOnMain	True to runs on main.
		//@param [in,out]	fn		  	The function.
		//@param 		 	args	  	Variable arguments providing [in,out] The arguments.

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
				[&, holdLock, holdSize, holdConditional, priority, runsOnMain]() mutable {

				//prefix code
				detail::InitPointers();

				//yielding garuntees that the fiber properties are saved at the expense of another round of context switching
				boost::this_fiber::properties< detail::FiberProperties >().SetPriority(priority, runsOnMain);
				boost::this_fiber::yield();

				//assert that the function is executing on the right thread
				assert(!boost::this_fiber::properties< detail::FiberProperties >().RunOnMain() ||
					(boost::this_fiber::properties< detail::FiberProperties >().RunOnMain() && detail::mainID == std::this_thread::get_id()));

#endif

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
			fiber.detach();

#endif

		}
		else {

#ifndef	SOUL_SINGLE_STACK

			boost::fibers::fiber fiber(
				[&, priority, runsOnMain]() mutable {

				//prefix code
				detail::InitPointers();

				//yielding garuntees that the fiber properties are saved at the expense of another round of context switching
				boost::this_fiber::properties< detail::FiberProperties >().SetPriority(priority, runsOnMain);
				boost::this_fiber::yield();

				//assert that the function is executing on the right thread
				assert(!boost::this_fiber::properties< detail::FiberProperties >().RunOnMain() ||
					(boost::this_fiber::properties< detail::FiberProperties >().RunOnMain() && detail::mainID == std::this_thread::get_id()));

#endif

				///////////////////////////////////////////
				fn(std::forward<Args>(args)...);
				///////////////////////////////////////////

				//suffix code
				detail::fiberMutex.lock();
				detail::fiberCount--;
				detail::fiberMutex.unlock();

#ifndef	SOUL_SINGLE_STACK

			});
			fiber.detach();

#endif

		}
	}

	//Blocks the fiber until all tasks with the LAUNCH_IMMEDIATE policy have been executed.
	void Block();

	//Yields the current fiber to the scheduler.
	void Defer();

	//---------------------------------------------------------------------------------------------------
	//Returns the running state of the scheduler. Useful for functions that want to run the
	//lifespan of the engine.
	//@return	True if it succeeds, false if it fails.

	bool Running();
};
