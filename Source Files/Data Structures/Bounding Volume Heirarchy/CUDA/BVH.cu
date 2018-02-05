#include "BVH.cuh"

#include "Compute\DeviceAPI.h"
#include "Utility/Includes/GLMIncludes.h"

// Returns the highest differing bit of i and i+1
__device__ uint HighestBit(uint i, uint64* morton)
{
	return morton[i] ^ morton[i + 1];
}

__global__ void BuildTree(uint n, BVHData* data, InnerNode* innerNodes, LeafNode* leafNodes, uint64* mortonCodes, BoundingBox* boxes)
{
	const uint index = ThreadIndex1D();

	if (index >= n) {
		return;
	}


	uint nodeID;
	const uint innerSize = n - 1;

	//first process the leaf parents
	if (index == 0 || index < innerSize &&
		HighestBit(index - 1, mortonCodes) > HighestBit(index, mortonCodes))
	{

		// parent = right, set parent left child and range to node	
		const uint parentID = index;
		InnerNode& parentNode = innerNodes[parentID];

		parentNode.childLeft = index;
		parentNode.rangeLeft = index;
		parentNode.leftLeaf = true;
		nodeID = parentID;

	}
	else
	{

		// parent = left -1, set parent right child and range to node
		const uint parentID = index - 1;
		InnerNode& parentNode = innerNodes[parentID];

		parentNode.childRight = index;
		parentNode.rangeRight = index;
		parentNode.rightLeaf = true;
		nodeID = parentID;

	}


	//next iterate until the root is hit
	while (true) {

		//only process first thread at the node
		if (atomicAdd(&innerNodes[nodeID].atomic, 1) != 1) {
			return;
		}

		//only one thread is here, read in the node
		InnerNode node = innerNodes[nodeID];

		//TODO store bounding boxes to cut 2 global reads
		//combine the bounding boxes
		const BoundingBox boxLeft = node.leftLeaf ? boxes[leafNodes[node.childLeft].boxID] : innerNodes[node.childLeft].box;
		const BoundingBox boxRight = node.rightLeaf ? boxes[leafNodes[node.childRight].boxID] : innerNodes[node.childRight].box;

		node.box.max = glm::max(boxLeft.max, boxRight.max);
		node.box.min = glm::min(boxLeft.min, boxRight.min);

		innerNodes[nodeID].box = node.box;

		if (node.rangeLeft == 0 && node.rangeRight == innerSize) {
			data->root = nodeID;
			return;
		}

		if (node.rangeLeft == 0 || node.rangeRight < innerSize &&
			HighestBit(node.rangeLeft - 1, mortonCodes) > HighestBit(node.rangeRight, mortonCodes))
		{

			// parent = right, set parent left child and range to node		
			const uint parentID = node.rangeRight;
			InnerNode& parentNode = innerNodes[parentID];
			
			parentNode.childLeft = nodeID;
			parentNode.rangeLeft = node.rangeLeft;
			nodeID = parentID;

		}
		else
		{

			// parent = left -1, set parent right child and range to node
			const uint parentID = node.rangeLeft - 1;
			InnerNode& parentNode = innerNodes[parentID];
			
			parentNode.childRight = nodeID;
			parentNode.rangeRight = node.rangeRight;
			nodeID = parentID;

		}

	}
}

__global__ void Reset(uint n, InnerNode* innerNodes, LeafNode* leafNodes)
{

	const uint index = ThreadIndex1D();

	if (index >= n) {
		return;
	}

	//inner node
	InnerNode tempInner;
	tempInner.atomic = 0; //inner nodes are not visited
	tempInner.leftLeaf = false;
	tempInner.rightLeaf = false;
	innerNodes[index] = tempInner;

	//TODO change 'index' to actual values of the data
	//set the leaf node
	LeafNode tempLeaf;
	tempLeaf.dataID = index;
	tempLeaf.boxID = index;
	leafNodes[index] = tempLeaf;
}