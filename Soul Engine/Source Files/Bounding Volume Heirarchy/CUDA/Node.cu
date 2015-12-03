#include "Node.cuh"

Node::Node(){

	systemMin = -8192.0f;
	systemMax = 8192.0f;

	box = NULL;

	childLeft = NULL;
	childRight = NULL;
}
Node::~Node(){

}