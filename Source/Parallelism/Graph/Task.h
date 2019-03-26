#pragma once

#include "Node.h"
#include <functional>

class FiberScheduler;

class Task : public Node{

public:

	Task(FiberScheduler*, std::function<void()>&&) noexcept;

	~Task() override = default;

	Task(const Task&) = delete;
	Task(Task&&) = default;

	Task& operator=(const Task&) = delete;
	Task& operator=(Task&&) = default;

	void Execute(std::chrono::nanoseconds) override;


private:

	FiberScheduler* scheduler_;
	std::function<void()> callable_;


};
