#include "BVH.cuh"

#include "Compute\DeviceAPI.h"
#include "Utility/Includes/GLMIncludes.h"

// Returns the highest differing bit of i and i+1
__device__ uint HighestBit(uint i, uint64* morton)
{
	return morton[i] ^ morton[i + 1];
}

__global__ void BuildTree(uint n, uint innerSize, BVHData* data, Node* nodes, uint64* mortonCodes)
{
	const uint index = ThreadIndex1D();
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