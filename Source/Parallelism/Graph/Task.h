#pragma once

#include "Node.h"
#include "Parallelism/Fiber/FiberParameters.h"

#include <functional>

class Scheduler;

class Task : public Node{

	//type-erased, no parameter, callable
	using FuncType = std::function<void()>;

public:

	Task(Scheduler*, FuncType&&) noexcept;

	~Task() override = default;

	Task(const Task&) = delete;
	Task(Task&& o) = default;

	Task& operator=(const Task&) = delete;
	Task& operator=(Task&& other) = default;

	void Execute() override;


private:

	Scheduler* scheduler_;

	FiberParameters parameters_;
	FuncType callable_;


};
