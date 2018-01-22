#include "BVH.cuh"

#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility/Includes/GLMIncludes.h"

// Returns the highest differing bit of i and i+1
__device__ uint HighestBit(uint i, uint64* morton)
{
	return morton[i] ^ morton[i + 1];
}

__global__ void BuildTree(uint n, BVHData* data, Node* nodes, uint64* mortonCodes, uint leafOffset)
{
	const uint index = getGlobalIdx_1D_1D();
	if (index >= n) {
		return;
	}

	Node* currentNode = nodes + (leafOffset + index);

	while (true) {
		// Allow only one thread to process a node
		if (atomicAdd(&(currentNode->atomic), 1) != 1)
			return;

		// Set bounding box if the node is no leaf
		if (currentNode - nodes < leafOffset)
		{
			currentNode->box.max = glm::max(currentNode->childLeft->box.max, currentNode->childRight->box.max);
			currentNode->box.min = glm::min(currentNode->childLeft->box.min, currentNode->childRight->box.min);
		}

		uint left = currentNode->rangeLeft;
		uint right = currentNode->rangeRight;

		if (left == 0 && right == leafOffset) {
			data->root = currentNode;
			return;
		}

		Node* parent;
		if (left == 0 || (right < leafOffset && HighestBit(left - 1, mortonCodes) > HighestBit(right, mortonCodes)))
		{
			// parent = right, set parent left child and range to node
			parent = nodes + right;
			parent->childLeft = currentNode;
			parent->rangeLeft = left;

		}
		else
		{
			// parent = left -1, set parent right child and range to node
			parent = nodes + (left - 1);
			parent->childRight = currentNode;
			parent->rangeRight = right;
		}

		currentNode = parent;
	}
}


__global__ void Reset(uint n, Node* nodes, Face* faces, Vertex* vertices, uint64* mortonCodes, uint leafOffset)
{
	const uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	Node temp;

	// Set ranges
	temp.rangeLeft = index;
	temp.rangeRight = index;
	temp.atomic = 1; // To allow the next thread to process
	temp.childLeft = nullptr;
	temp.childRight = nullptr;

	if (index < leafOffset) {
		Node tempF;

		tempF.atomic = 0;
		tempF.childLeft = nodes + leafOffset + index;
		tempF.childRight = nodes + leafOffset + index + 1;

		nodes[index] = tempF;
	}

	// Set triangles in leaf
	temp.faceID = index;

	glm::uvec3 ind = faces[index].indices;

	// Expand bounds using min/max functions
	glm::vec3 pos0 = vertices[ind.x].position;
	glm::vec3 max = pos0;
	glm::vec3 min = pos0;

	glm::vec3 pos1 = vertices[ind.y].position;
	max = glm::max(pos1, max);
	min = glm::min(pos1, min);

	glm::vec3 pos2 = vertices[ind.z].position;
	max = glm::max(pos2, max);
	min = glm::min(pos2, min);

	temp.box.max = max;
	temp.box.min = min;

	nodes[leafOffset + index] = temp;

	// Special case
	if (n == 1)
	{
		nodes[0].box = nodes[leafOffset + 0].box;
		nodes[0].childLeft = nodes + leafOffset + 0;
	}
}