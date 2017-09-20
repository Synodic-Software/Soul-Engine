#include "stdafx.h"
#include "CppUnitTest.h"
#include <boost/fiber/all.hpp>
#include "Source Files/Multithreading/Scheduler.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace Multithreadingtest
{		
	TEST_CLASS(unittest1)
	{
	public:
		
		TEST_METHOD(fiber_spawning)
		{
			Assert::IsTrue(true);
		}

	};
}