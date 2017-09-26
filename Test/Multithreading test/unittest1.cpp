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
			//tests intitializing, adding a simple task and terminating the scheduler
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
			//tests spawning 800 fibers of every combination of boolean/enums
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
				Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, []() {
					mult(1, 1);
				});
				Scheduler::AddTask(LAUNCH_CONTINUE, FIBER_HIGH, true, []() {
					mult(1, 1);
				});
				Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_LOW, true, []() {
					mult(1, 1);
				});
				Scheduler::AddTask(LAUNCH_CONTINUE, FIBER_LOW, true, []() {
					mult(1, 1);
				});
			}
			

			Assert::IsTrue(Scheduler::Running());

			Scheduler::Block();

			Scheduler::Terminate();

			Assert::IsTrue(true);
		}
		TEST_METHOD(blocker)
		{
			//Tests blocking of execution until all launch immediate tasks are done
			//Also ensures that blockage doesn't go into effect when their is no launch immediate
			Scheduler::Initialize();

			//due to block each task runs sequentially so a is in order
			std::vector<int> a;
			for (size_t i = 0; i < 100; i++)
			{
				Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, [&i, &a]() {
					a.push_back(i);
				});
				Scheduler::Block();
			}

			//since block has no effect b is out of order
			std::vector<int> b;
			for (size_t i = 0; i < 100; i++)
			{
				Scheduler::AddTask(LAUNCH_CONTINUE, FIBER_HIGH, false, [&i, &b]() {
					b.push_back(i);
				});
				Scheduler::Block();
			}

			Scheduler::Terminate();
			for (size_t i = 0; i < 100; i++)
			{
				Assert::AreEqual(int(i), a[i]);
			}
			bool out_of_order = true;
			for (size_t i = 1; i < 100; i++)
			{
				out_of_order = out_of_order && b[i - 1] < b[i];
			}
			Assert::IsFalse(out_of_order);


			Assert::IsTrue(true);
		}
	};
}