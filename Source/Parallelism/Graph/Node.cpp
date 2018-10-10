#include "Node.h"

#include "Graph.h"
#include "Task.h"

void Node::DependsOn(Task& other) {

	other.AddChild(this);

}

void Node::DependsOn(Graph& other) {

	other.AddChild(this);

}

void Node::AddChild(Node* child) {

	children_.push_back(child);

}