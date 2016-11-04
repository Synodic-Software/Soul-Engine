#include "Scheduler.h"
#include <boost/fiber/all.hpp>

using namespace boost::fibers;
using namespace boost::this_fiber;


//let's do something with this 

//#include "fiber_tasking_lib/tagged_heap_backed_linear_allocator.h"


//FiberTaskingLib::TaskScheduler *taskScheduler = NULL;
//FiberTaskingLib::TaggedHeap *taggedHeap = NULL;
//FiberTaskingLib::TaggedHeapBackedLinearAllocator *allocator = NULL;

//// new task submitted to scheduler
//void Scheduler::Start(TASK task) {
//	
//	taskScheduler = new FiberTaskingLib::TaskScheduler();
//	taskScheduler->Initialize(10);
//
//	taggedHeap = new FiberTaskingLib::TaggedHeap(2097152);
//	allocator = new FiberTaskingLib::TaggedHeapBackedLinearAllocator();
//	allocator->init(taggedHeap, 1234);
//	
//	std::shared_ptr<FiberTaskingLib::AtomicCounter> counter = taskScheduler->AddTasks(1, &task);
//
//	taskScheduler->WaitForCounter(counter, 0);
//}

//// end task
//void Scheduler::Terminate() {
//	if (allocator != NULL){
//		taskScheduler->Quit();
//		allocator->destroy();
//		delete allocator;
//		delete taggedHeap;
//		delete taskScheduler;
//	}
//}