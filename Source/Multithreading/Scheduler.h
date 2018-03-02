#pragma once

#include "FiberProperties.h"

#include <boost/fiber/fss.hpp>
#include <boost/fiber/condition_variable.hpp>

#include <thread>
#include <mutex>

//Boost fiber includes some nasty windows stuff
#undef CreateWindow


//TODO: Implement boost Fiber: Work_stealing



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

class Scheduler {

public:

	static Scheduler& Instance()
	{
		static Scheduler instance;
		return instance;
	}

	Scheduler(Scheduler const&) = delete;
	void operator=(Scheduler const&) = delete;


	template <typename T, typename ... Args>
	void AddTask(FiberPolicy, FiberPriority, bool, T*, void(T::*)(Args...) const);

	template <typename T, typename ... Args>
	void AddTask(FiberPolicy, FiberPriority, bool, T*, void(T::*)(Args...));

	template<typename Fn, typename ... Args>
	void AddTask(FiberPolicy, FiberPriority, bool, Fn &&, Args && ...);


	void Block();
	void Defer();
	bool Running() const;

private:

	Scheduler();
	~Scheduler();

	void InitPointers();
	void ThreadRun();

	std::thread::id mainID;

	std::size_t fiberCount;
	std::mutex fiberMutex;

	bool shouldRun;

	boost::fibers::fiber_specific_ptr<std::size_t> holdCount;
	boost::fibers::fiber_specific_ptr<std::mutex> holdMutex;
	boost::fibers::fiber_specific_ptr<boost::fibers::condition_variable_any> blockCondition;

	std::vector<std::thread> threads;
	boost::fibers::condition_variable_any threadCondition;

};


/*
*    Adds a task.
*    @param 		 	policy	  	The policy.
*    @param 		 	priority  	The priority.
*    @param 		 	runsOnMain	True to runs on main.
*    @param [in,out]	instance  	If non-null, the instance.
*    @param [in,out]	func	  	If non-null, the function.
*/

template <typename T, typename ... Args>
void Scheduler::AddTask(FiberPolicy policy, FiberPriority priority, bool runsOnMain, T *instance, void(T::*func)(Args...) const) {
	AddTask(policy, priority, runsOnMain, [=](Args... args) {
		(instance->*func)(args...);
	});
}

template <typename T, typename ... Args>
void Scheduler::AddTask(FiberPolicy policy, FiberPriority priority, bool runsOnMain, T *instance, void(T::*func)(Args...)) {
	AddTask(policy, priority, runsOnMain, [=](Args... args) {
		(instance->*func)(args...);
	});
}


/*
*    Adds a task.
*    @param 		 	policy	  	The fiber policy after running the segment
*    @param 		 	priority  	Fiber execution priority
*    @param 		 	runsOnMain	Requirement that this function runs on the main thread
*    @param [in,out]	fn		  	The function.
*    @param 		 	args	  	Variable arguments providing [in,out] The arguments.
*	 @
*/

template<typename Fn, typename ... Args>
void Scheduler::AddTask(FiberPolicy policy, FiberPriority priority, bool runsOnMain, Fn && fn, Args && ... args) {

	//this thread increments the locks, the launched fiber implements the decrement
	fiberMutex.lock();
	fiberCount++;
	fiberMutex.unlock();

	auto fiberExecute = [mainID, priority, runsOnMain]() mutable {
		//prefix code
		InitPointers();

		//yielding garuntees that the fiber properties are saved at the expense of another round of context switching
		boost::this_fiber::properties< FiberProperties >().SetPriority(mainID, priority, runsOnMain);
		boost::this_fiber::yield();

		//assert that the function is executing on the right thread
		assert(!boost::this_fiber::properties< FiberProperties >().RunOnMain() ||
			boost::this_fiber::properties< FiberProperties >().RunOnMain() && mainID == std::this_thread::get_id());


		///////////////////////////////////////////
		fn(std::forward<Args>(args)...);
		///////////////////////////////////////////

		//suffix code
		fiberMutex.lock();
		fiberCount--;
		fiberMutex.unlock();
	};

	//only difference is the hold lock increment
	if (policy == LAUNCH_IMMEDIATE) {

		std::mutex* holdLock = holdMutex.get();
		std::size_t* holdSize = holdCount.get();
		boost::fibers::condition_variable_any* holdConditional = blockCondition.get();

		holdLock->lock();
		(*holdSize)++;
		holdLock->unlock();


		//lambda wrapping the called function with other information
		boost::fibers::fiber fiber(
			[holdLock, holdSize, holdConditional]() mutable {

			fiberExecute();

			holdLock->lock();
			(*holdSize)--;
			holdLock->unlock();
			holdConditional->notify_all();

		});
		fiber.detach();

	}
	else {

		boost::fibers::fiber fiber(
			fiberExecute()
		);
		fiber.detach();

	}
}