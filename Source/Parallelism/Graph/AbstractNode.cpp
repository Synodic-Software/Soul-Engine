#include "Task.h"

Task::Task(FuncType&& callable) :
	callable_(std::forward<FuncType>(callable))
{

}