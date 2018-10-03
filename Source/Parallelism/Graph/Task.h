#pragma once

#include "AbstractNode.h"
#include "Parallelism/Fiber/FiberParameters.h"

#include <functional>

class Task : public AbstractNode {

	//type-erased, no parameter, callable
	using FuncType = std::function<void()>;

public:

	Task(FuncType&&);

	~Task() = default;

	Task(const Task&) = delete;
	Task(Task&& o) noexcept = delete;

	Task& operator=(const Task&) = delete;
	Task& operator=(Task&& other) noexcept = delete;

	void Execute(Scheduler&) const;

private:

	FiberParameters parameters_;
	FuncType callable_;

};
