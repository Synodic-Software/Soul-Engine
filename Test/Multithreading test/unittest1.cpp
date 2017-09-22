#include "stdafx.h"
#include "CppUnitTest.h"
#include <boost/fiber/all.hpp>
#include <Source Files\Multithreading\Scheduler.h>
//#include "Source Files/Multithreading/Scheduler.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

void mult(int x, int y) {
	int z = x*y;
}

namespace Multithreadingtest
{		
	TEST_CLASS(unittest1)
	{
	public:
		
		TEST_METHOD(basic_fiber_spawning)
		{
			//test intitializing the scheduler
			Scheduler::Initialize();


			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
				std::cout << "hi" << std::endl;
			});

			Assert::IsTrue(Scheduler::Running());

			Scheduler::Block();

			Scheduler::Terminate();

			Assert::IsTrue(true);
		}
		TEST_METHOD(Multi_fiber_spawning)
		{
			//test intitializing the scheduler
			Scheduler::Initialize();

			for (UINT i = 0; i < 100; i++)
			{
				Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
					mult(1,1);
				});
				Scheduler::AddTask(LAUNCH_CONTINUE, FIBER_HIGH, false, []() {
					mult(1, 1);
				});
				Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_LOW, false, []() {
					mult(1, 1);
				});
				Scheduler::AddTask(LAUNCH_CONTINUE, FIBER_LOW, false, []() {
					mult(1, 1);
				});

			}
			

			Assert::IsTrue(Scheduler::Running());

			Scheduler::Block();

			Scheduler::Terminate();

			Assert::IsTrue(true);
		}

	};
}