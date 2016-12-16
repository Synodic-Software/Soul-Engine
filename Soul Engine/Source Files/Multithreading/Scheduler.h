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

		template< typename Fn, typename ... Fns >
		void WaitAllHelper(boost::fibers::launch policy, std::shared_ptr< boost::fibers::barrier > barrier,
			Fn && function, Fns && ... functions) {
			boost::fibers::fiber(
				policy,
				std::bind(
					[](std::shared_ptr< boost::fibers::barrier > & barrier,
						typename std::decay< Fn >::type & function) mutable {
				function();
				barrier->wait();
			},
					barrier,
				std::forward< Fn >(function)
				)).detach();
			WaitAllHelper(barrier, std::forward< Fns >(functions) ...);
		}


		template< typename Fn, typename ... Fns >
		void RunAllHelper(boost::fibers::launch policy,
			Fn && function, Fns && ... functions) {
			boost::fibers::fiber(
				policy,
				std::bind(
					[](std::shared_ptr< boost::fibers::barrier > & barrier,
						typename std::decay< Fn >::type & function) mutable {
				function();
				barrier->wait();
			},
					barrier,
				std::forward< Fn >(function)
				)).detach();
			RunAllHelper(barrier, std::forward< Fns >(functions) ...);
		}

	}


	void Init();
	void Terminate();

	//Add a list of tasks 
	template< typename ... Fns >
	void AddTasks(FiberPolicy policy, Fns && ... functions) {
		if (policy == IMMEDIATE) {
			std::size_t count(sizeof ... (functions));
			detail::fiberCount += count;
			auto barrier(std::make_shared< boost::fibers::barrier >(count + 1));
			detail::WaitAllHelper((boost::fibers::launch)policy, barrier, std::forward< Fns >(functions) ...);
			barrier->wait();
			detail::fiberCount -= count;
		}
		else {


		}
	}



	//ex

	//WaitAll(
	//	[]() { sleeper("was_long", 150); },
	//	[]() { sleeper("was_medium", 100); },
	//	[]() { sleeper("was_short", 50); });

	template<typename Fn,
		typename ... Args>
		void AddTask(FiberPolicy policy, Fn && fn, Args && ... args) {

		boost::fibers::fiber((boost::fibers::launch)policy, std::forward< Fn >(fn), std::forward< Args >(args) ...).detach();
		detail::fiberCount++;

	}
};
