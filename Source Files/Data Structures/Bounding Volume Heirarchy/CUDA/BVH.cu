#include "BVH.cuh"

#include "Compute\DeviceAPI.h"
#include "Utility/Includes/GLMIncludes.h"

// Returns the highest differing bit of i and i+1
__device__ uint HighestBit(uint i, uint64* morton)
{
	return morton[i] ^ morton[i + 1];
}

__global__ void BuildTree(uint n, BVHData* data, Node* nodes, uint64* mortonCodes, BoundingBox* boxes)
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
		Node& parentNode = nodes[parentID];

		parentNode.childLeft = index + n;
		parentNode.rangeLeft = index;
		nodeID = parentID;

	}
	else
	{

		// parent = left -1, set parent right child and range to node
		const uint parentID = index - 1;
		Node& parentNode = nodes[parentID];

		parentNode.childRight = index + n;
		parentNode.rangeRight = index;
		nodeID = parentID;

	}


	//next iterate until the root is hit
	while (true) {

		//only process first thread at the node
		if (atomicAdd(&nodes[nodeID].atomic, 1) != 1) {
			return;
		}

		//only one thread is here, read in the node
		Node node = nodes[nodeID];

		//TODO store bounding boxes to cut 2 global reads
		//combine the bounding boxes
		const BoundingBox boxLeft = node.childLeft >= n ? boxes[node.childLeft - n] : nodes[node.childLeft].box;
		const BoundingBox boxRight = node.childRight >= n ? boxes[node.childRight - n] : nodes[node.childRight].box;

		node.box.max = glm::max(boxLeft.max, boxRight.max);
		node.box.min = glm::min(boxLeft.min, boxRight.min);

		nodes[nodeID].box = node.box;

		if (node.rangeLeft == 0 && node.rangeRight == innerSize) {
			data->root = nodeID;
			return;
		}

		if (node.rangeLeft == 0 || node.rangeRight < innerSize &&
			HighestBit(node.rangeLeft - 1, mortonCodes) > HighestBit(node.rangeRight, mortonCodes))
		{

			// parent = right, set parent left child and range to node		
			const uint parentID = node.rangeRight;
			Node& parentNode = nodes[parentID];
			
			parentNode.childLeft = nodeID;
			parentNode.rangeLeft = node.rangeLeft;
			nodeID = parentID;

		}
		else
		{

			// parent = left -1, set parent right child and range to node
			const uint parentID = node.rangeLeft - 1;
			Node& parentNode = nodes[parentID];
			
			parentNode.childRight = nodeID;
			parentNode.rangeRight = node.rangeRight;
			nodeID = parentID;

		}

	}
}

__global__ void Reset(uint n, Node* nodes)
{

	const uint index = ThreadIndex1D();

	if (index >= n) {
		return;
	}

	//inner node
	Node tempInner;
	tempInner.atomic = 0; //inner nodes are not visited
	nodes[index] = tempInner;

}