#pragma once

#include "Parallelism/Modules/Fiber/FiberParameters.h"

#include <vector>
#include <chrono>

class Task;
class Graph;

//Handles the topology of the graph
class Node{

public:

	Node();
	virtual ~Node() = default;

	Node(const Node&) = delete;
	Node(Node&&) noexcept = default;

	Node& operator=(const Node&) = delete;
	Node& operator=(Node&&) noexcept = default;

	virtual void Execute(std::chrono::nanoseconds = std::chrono::nanoseconds(0)) = 0;

	void DependsOn(Task&);
	void DependsOn(Graph&);

	bool Root();
	void Root(bool);

protected:

	FiberParameters parameters_;
	std::vector<Node*> children_;


private:

	void AddChild(Node*);

	bool root;

};
