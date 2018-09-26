#pragma once

#include "Task.h"

#include <forward_list>
#include <functional>

class Graph{

public:

	Graph() = default;
	~Graph() = default;

	Graph(const Graph&) = delete;
	Graph(Graph&& o) noexcept = delete;

	Graph& operator=(const Graph&) = delete;
	Graph& operator=(Graph&& other) noexcept = delete;

	template <typename Callable>
	auto Emplace(Callable&&);


private:

	std::forward_list<Task<std::function>> tasks_;
	std::forward_list<Graph> subGraphs_;

};

template <typename Callable>
auto Graph::Emplace(Callable&& c) {

}