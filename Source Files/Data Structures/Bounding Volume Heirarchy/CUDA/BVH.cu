#include "BVH.cuh"

#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility/Includes/GLMIncludes.h"

// Returns the highest differing bit of i and i+1
__device__ uint HighestBit(uint i, uint64* morton)
{
	return morton[i] ^ morton[i + 1];
}

__global__ void BuildTree(uint n, uint innerSize, BVHData* data, Node* nodes, uint64* mortonCodes)
{
	const uint index = getGlobalIdx_1D_1D();
	if (index >= n) {
		return;
	}

	uint nodeOffset = innerSize + index;
	Node* nodePointer = nodes + nodeOffset;
	Node currentNode = nodes[nodeOffset];
	
	while (true) {
		// Allow only one thread to process a node
		if (atomicAdd(&nodePointer->atomic, 1) != 1)
			return;

		// Set bounding box if the node is not a leaf
		if (nodeOffset < innerSize)
		{
			const BoundingBox boxLeft = nodes[currentNode.childLeft].box;
			const BoundingBox boxRight = nodes[currentNode.childRight].box;

			currentNode.box.max = glm::max(boxLeft.max, boxRight.max);
			currentNode.box.min = glm::min(boxLeft.min, boxRight.min);

			nodes[nodeOffset] = currentNode;
		}

		if (currentNode.rangeLeft == 0 && currentNode.rangeRight == innerSize) {
			data->root = nodeOffset;
			return;
		}

		Node* parentPointer;

		if (currentNode.rangeLeft == 0 || currentNode.rangeRight < innerSize && 
			HighestBit(currentNode.rangeLeft - 1, mortonCodes) > HighestBit(currentNode.rangeRight, mortonCodes))
		{

			// parent = right, set parent left child and range to node		
			parentPointer = nodes + currentNode.rangeRight;
			Node parent = *parentPointer;
			parent.childLeft = nodeOffset;
			parent.rangeLeft = currentNode.rangeLeft;
			*parentPointer = parent;

		}
		else
		{

			// parent = left -1, set parent right child and range to node
			parentPointer = nodes + (currentNode.rangeLeft - 1);
			Node parent = *parentPointer;
			parent.childRight = nodeOffset;
			parent.rangeRight = currentNode.rangeRight;
			*parentPointer = parent;
			
		}

		nodePointer = parentPointer;
		nodeOffset =  nodePointer - nodes;
		currentNode = *nodePointer;
	}
}


__global__ void Reset(uint n, uint innerSize, Node* nodes, Face* faces, Vertex* vertices)
{
	const uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}


	const uint leafOffset = innerSize + index;

	//set the inner node
	if (index < innerSize) {
		Node temp;
		temp.atomic = 0; //inner nodes are not visited
		temp.childLeft = leafOffset;
		temp.childRight = leafOffset + 1;
		nodes[index] = temp;
	}


	const glm::uvec3 ind = faces[index].indices;

	// Expand bounds using min/max functions
	const glm::vec3 pos0 = vertices[ind.x].position;
	const glm::vec3 pos1 = vertices[ind.y].position;
	const glm::vec3 pos2 = vertices[ind.z].position;

	glm::vec3 max = pos0;
	glm::vec3 min = pos0;

	max = glm::max(pos1, max);
	min = glm::min(pos1, min);

	max = glm::max(pos2, max);
	min = glm::min(pos2, min);

	//set the leaf node
	Node temp;
	temp.rangeLeft = index;
	temp.rangeRight = index;
	temp.atomic = 1; // To allow the next thread to process
	temp.faceID = index; //set triangle
	temp.box.max = max;
	temp.box.min = min;
	temp.childLeft = static_cast<uint>(-1); //set termination
	temp.childRight = static_cast<uint>(-1); //set termination

	nodes[leafOffset] = temp;
}