#include "Node.cuh"

Node::Node(){

	box = NULL;

	childLeft = NULL;
	childRight = NULL;
}
Node::~Node(){

}

Node* Node::GetRight(){
	return childRight;
}

Node* Node::GetLeft(){
	return childLeft;
}