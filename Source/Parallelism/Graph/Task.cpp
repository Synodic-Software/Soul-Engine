#include "Task.h"

#include "Parallelism/Fiber/Scheduler.h"

Task::Task(Scheduler* scheduler, FuncType&& callable) noexcept:
	scheduler_(scheduler),
	parameters_(),
	callable_(std::forward<FuncType>(callable))
{

}

void Task::Execute() {
	
	scheduler_->AddTask(parameters_, [this]()
	{
		std::invoke(std::forward<FuncType>(callable_));

		for (const auto& child : children_) {

			child->Execute();

		}

		scheduler_->Block();
	});

}