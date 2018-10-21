#pragma once

#include <vector>

class Task;
class Graph;

//Handles the topology of the graph
class Node{

public:

	Node() = default;
	virtual ~Node() = default;

	Node(const Node&) = delete;
	Node(Node&& o) noexcept = default;

	Node& operator=(const Node&) = delete;
	Node& operator=(Node&& other) noexcept = default;

	virtual void Execute() = 0;

	void DependsOn(Task&);
	void DependsOn(Graph&);


protected:

	std::vector<Node*> children_;

private:

	void AddChild(Node*);

};
