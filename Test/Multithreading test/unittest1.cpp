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
		
		TEST_METHOD(fiber_spawning)
		{
			//test intitializing the scheduler
			Scheduler::Initialize();


			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, []() {
				std::cout << "hi" << std::endl;
			});

			Scheduler::Block();

			Scheduler::Terminate();

			Assert::IsTrue(true);
		}

	};
}