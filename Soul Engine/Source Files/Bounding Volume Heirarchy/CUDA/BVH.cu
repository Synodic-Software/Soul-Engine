#include "BVH.cuh"

BVH::BVH( Face*** datan, uint64** mortonCodesn){
	data = datan;
	mortonCodes = mortonCodesn;
	currentSize = 0;
	allocatedSize = 0;
}

// Returns the highest differing bit of i and i+1
__device__ uint HighestBit(uint i, uint64* morton)
{
	return morton[i] ^ morton[i + 1];
}


// Sets the bounding box and traverses to root
__device__ void ProcessParent(const uint nData, Node* currentNode, Node* nodes, uint64* morton, const uint leafOffset)
{
	// Allow only one thread to process a node
	if (atomicAdd(&(currentNode->atomic), 1) != 1)
		return;

	// Set bounding box if the node is no leaf
	if (currentNode - nodes<leafOffset)
	{
		//update the node's bounding volume
		glm::vec3 max = currentNode->childLeft->box.origin + currentNode->childLeft->box.extent;
		glm::vec3 min = currentNode->childLeft->box.origin - currentNode->childLeft->box.extent;

		glm::vec3 objMax = currentNode->childRight->box.origin + currentNode->childRight->box.extent;
		glm::vec3 objMin = currentNode->childRight->box.origin - currentNode->childRight->box.extent;

		glm::vec3 newMax = glm::max(max, objMax);
		glm::vec3 newMin = glm::min(min, objMin);

		currentNode->box.origin = ((newMax - newMin) / 2.0f) + newMin;
		currentNode->box.extent = currentNode->box.origin - newMin;
	}

	uint left = currentNode->rangeLeft;
	uint right = currentNode->rangeRight;

	Node* parent;
	if (left == 0 || (right != nData - 1 && HighestBit(right, morton) < HighestBit(left - 1, morton)))
	{
		// parent = right, set parent left child and range to node
		parent = nodes+right;
		parent->childLeft = currentNode;
		parent->rangeLeft = left;

	}
	else
	{
		// parent = left -1, set parent right child and range to node
		parent = nodes+(left - 1);
		parent->childRight = currentNode;
		parent->rangeRight = right;
	}

	if (left == 0 && right == nData){
		return;
	}
	ProcessParent(nData, parent, nodes, morton, leafOffset);
}

__global__ void BuildTree(const uint n, Node* nodes, Face** data, uint64* mortonCodes, const uint leafOffset)
{
	uint index = getGlobalIdx_1D_1D();
	if (index >= n)
		return;

	ProcessParent(n, nodes + (leafOffset+index), nodes, mortonCodes, leafOffset);
}

__global__ void Reset(const uint n,Node* nodes, Face** data, uint64* mortonCodes,const uint leafOffset)
{
	uint index = getGlobalIdx_1D_1D();
	if (index >= n)
		return;

	// Reset parameters for internal and leaf nodes here

	// Set ranges
	nodes[leafOffset + index].rangeLeft = index;
	nodes[leafOffset + index].rangeRight = index;
	nodes[leafOffset + index].atomic = 1; // To allow the next thread to process
	if (index<leafOffset){
		nodes[index].atomic = 0; // Second thread to process
	}


	// Set triangles in leaf
	Face* face = *(data + index);
	nodes[leafOffset + index].faceID = face;

	// Expand bounds using min/max functions

	glm::vec3 max = face->objectPointer->vertices[face->indices.x].position;
	glm::vec3 min = face->objectPointer->vertices[face->indices.x].position;

	max = glm::max(face->objectPointer->vertices[face->indices.y].position, max);
	min = glm::min(face->objectPointer->vertices[face->indices.y].position, min);

	max = glm::max(face->objectPointer->vertices[face->indices.z].position, max);
	min = glm::min(face->objectPointer->vertices[face->indices.z].position, min);

	nodes[leafOffset + index].box.origin = ((max - min) / 2.0f) + min;
	nodes[leafOffset + index].box.extent = nodes[leafOffset + index].box.origin - min;

	// Special case
	if (n == 1)
	{
		nodes[0].box = nodes[leafOffset+0].box;
		nodes[0].childLeft = &nodes[leafOffset+0];
	}
}

void BVH::Build(uint size){
	currentSize = size;
	if (currentSize > allocatedSize){

		Node* nodeTemp;

		allocatedSize = glm::max(uint(allocatedSize * 1.5f), (currentSize * 2) - 1);


		cudaMallocManaged(&nodeTemp, allocatedSize * sizeof(Node));

		cudaFree(bvh);
		bvh = nodeTemp;
	}

	uint blockSize = 64;
	uint gridSize = (currentSize + blockSize - 1) / blockSize;

	CudaCheck(cudaDeviceSynchronize());

	Reset << <gridSize, blockSize >> >(currentSize, bvh, *data, *mortonCodes, currentSize - 1);
	CudaCheck(cudaDeviceSynchronize());

	BuildTree << <gridSize, blockSize >> >(currentSize, bvh, *data, *mortonCodes, currentSize - 1);

	CudaCheck(cudaDeviceSynchronize());

}