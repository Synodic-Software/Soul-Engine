#pragma once

#include "Node.h"
#include "Parallelism/Fiber/FiberParameters.h"
#include <functional>

class Scheduler;

class Task : public Node{

public:

	Task(Scheduler*, std::function<void()>&&) noexcept;

	~Task() override = default;

	Task(const Task&) = delete;
	Task(Task&&) = default;

	Task& operator=(const Task&) = delete;
	Task& operator=(Task&&) = default;

	void Execute(std::chrono::nanoseconds) override;


private:

	Scheduler* scheduler_;
	std::function<void()> callable_;


};
