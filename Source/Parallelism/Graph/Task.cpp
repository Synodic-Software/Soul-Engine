#include "Task.h"

#include "Parallelism/Fiber/Scheduler.h"

Task::Task(FuncType&& callable) :
	parameters_(),
	callable_(std::forward<FuncType>(callable))
{

}

void Task::Execute(Scheduler& scheduler) const {
	
	scheduler.AddTask(parameters_, callable_);

}
