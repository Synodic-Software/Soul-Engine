#pragma once

#include "Task.h"
#include "AbstractNode.h"

#include "Core/Utility/Template/TypeTraits.h"

#include <forward_list>


class Scheduler;

class Graph : public AbstractNode{

public:

	Graph(Scheduler&);
	~Graph() = default;

	Graph(const Graph&) = delete;
	Graph(Graph&& o) noexcept = delete;

	Graph& operator=(const Graph&) = delete;
	Graph& operator=(Graph&& other) noexcept = delete;

	template <typename Callable>
	Task& Emplace(Callable&&);

	void Execute();

private:

	std::forward_list<Task> tasks_;

	Scheduler& scheduler_;

};

//Places the provided callable into the 
template <typename Callable>
Task& Graph::Emplace(Callable&& callable) {

	//return GraphProxy(tasks_).Emplace(std::forward<Callable>(callable));
	if constexpr (!std::is_invocable_v<Callable>) {

		static_assert(dependent_false_v<Callable>, "The provided parameter is not callable");

	}

	Task& task = tasks_.emplace_front(std::forward<Callable>(callable));

	return task;
}