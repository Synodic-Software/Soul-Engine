#include "Node.h"

#include "Graph.h"
#include "Task.h"

Node::Node():
	root(false)
{	
}

void Node::DependsOn(Task& other) {

	root = false;
	other.AddChild(this);

}

void Node::DependsOn(Graph& other) {

	root = false;
	other.AddChild(this);

}

bool Node::Root() {

	return root;

}

void Node::Root(bool other) {

	root = other;

}

void Node::AddChild(Node* child) {

	children_.push_back(child);

}