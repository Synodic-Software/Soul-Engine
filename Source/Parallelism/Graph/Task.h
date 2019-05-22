#pragma once

#include "Node.h"
#include <functional>
#include <memory>

class SchedulerModule;

class Task : public Node{

public:

	Task(std::shared_ptr<SchedulerModule>&, std::function<void()>&&) noexcept;

	~Task() override = default;

	Task(const Task&) = delete;
	Task(Task&&) = default;

	Task& operator=(const Task&) = delete;
	Task& operator=(Task&&) = default;

	void Execute(std::chrono::nanoseconds) override;


private:

	std::shared_ptr<SchedulerModule> scheduler_;
	std::function<void()> callable_;


};
