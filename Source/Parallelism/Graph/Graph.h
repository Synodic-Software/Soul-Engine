#pragma once

#include "Node.h"
#include "Task.h"

#include "Core/Utility/Template/TypeTraits.h"

#include <forward_list>

class SchedulerModule;

class Graph : public Node {


public:

	Graph(SchedulerModule*);

	~Graph() override = default;

	Graph(const Graph&) = delete;
	Graph(Graph&&) noexcept = default;

	Graph& operator=(const Graph&) = delete;
	Graph& operator=(Graph&&) noexcept = default;

	template <typename Callable>
	Task& AddTask(Callable&&);
	Graph& AddGraph();
	
	void Execute(std::chrono::nanoseconds) override;


private:

	SchedulerModule* scheduler_;

	std::forward_list<Task> tasks_;
	std::forward_list<Graph> graphs_;


};

//Create a tasks under this graph's control
template <typename Callable>
Task& Graph::AddTask(Callable&& callable) {

	if constexpr (!std::is_invocable_v<Callable>) {

		static_assert(dependent_false_v<Callable>, "The provided parameter is not callable");

	}

	Task& task = tasks_.emplace_front(scheduler_, std::forward<Callable>(callable));

	task.DependsOn(*this);
	task.Root(true);

	return task;

}