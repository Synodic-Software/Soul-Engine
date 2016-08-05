#include "Scheduler.h"
#include "fiber_tasking_lib/tagged_heap_backed_linear_allocator.h"


FiberTaskingLib::TaskScheduler *taskScheduler;
FiberTaskingLib::TaggedHeap *taggedHeap;
FiberTaskingLib::TaggedHeapBackedLinearAllocator *allocator;
// new task submitted to scheduler
void Scheduler::Start(TASK task) {
	
	taskScheduler = new FiberTaskingLib::TaskScheduler();
	taskScheduler->Initialize(110);

	taggedHeap = new FiberTaskingLib::TaggedHeap(2097152);
	allocator = new FiberTaskingLib::TaggedHeapBackedLinearAllocator();
	allocator->init(taggedHeap, 1234);
	
	std::shared_ptr<FiberTaskingLib::AtomicCounter> counter = taskScheduler->AddTasks(1, &task);

	//taskScheduler->WaitForCounter(counter, 0);
}

// end task
void Scheduler::Terminate() {
	
	taskScheduler->Quit();
	allocator->destroy();
	delete allocator;
	delete taggedHeap;
	delete taskScheduler;
}

