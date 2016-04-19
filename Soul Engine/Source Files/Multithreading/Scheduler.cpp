#include "Scheduler.h"

FiberTaskingLib::GlobalArgs *globalArgs;

// new task submitted to scheduler
void Scheduler::Start(TASK task) {
	
	globalArgs = new FiberTaskingLib::GlobalArgs();
	globalArgs->g_taskScheduler.Initialize(25, globalArgs);
	globalArgs->g_allocator.init(&globalArgs->g_heap, 1);
	
	std::shared_ptr<FiberTaskingLib::AtomicCounter> counter =globalArgs->g_taskScheduler.AddTasks(1, &task);

	globalArgs->g_taskScheduler.WaitForCounter(counter, 0);
}

// end task
void Scheduler::Terminate() {
	
	globalArgs->g_taskScheduler.Quit();
	globalArgs->g_allocator.destroy();
	delete globalArgs;
}

