#include "BVH.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility/Logger.h"

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

__global__ void BuildTree(const uint n, Node* nodes, Face** data, uint64* mortonCodes, const uint leafOffset, BVH* bvh)
{
	uint index = getGlobalIdx_1D_1D();
	if (index >= n)
		return;

	Node* currentNode = nodes + (leafOffset + index);

	while (true){
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

		if (left == 0 && right == leafOffset){
			bvh->root = currentNode;
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
	nodes[leafOffset + index].childLeft = NULL; // Second thread to process
	nodes[leafOffset + index].childRight = NULL; // Second thread to process
	if (index<leafOffset){
		/*nodes[index].rangeLeft = index;   //unneeded as all nodes are touched and updated
		nodes[index].rangeRight = index + 1;*/
		nodes[index].atomic = 0; // Second thread to process
		nodes[index].childLeft = &nodes[leafOffset + index]; // Second thread to process
		nodes[index].childRight = &nodes[leafOffset + index+1]; // Second thread to process
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

	nodes[leafOffset + index].box.max = max;
	nodes[leafOffset + index].box.min = min;

	// Special case
	if (n == 1)
	{
		nodes[0].box = nodes[leafOffset+0].box;
		nodes[0].childLeft = &nodes[leafOffset+0];
	}
}

void BVH::Build(uint size){
	cudaEvent_t start, stop;
	float time;
	CudaCheck(cudaEventCreate(&start));
	CudaCheck(cudaEventCreate(&stop));
	CudaCheck(cudaEventRecord(start, 0));

	currentSize = size;
	if (currentSize > allocatedSize){

		Node* nodeTemp;

		allocatedSize = glm::max(uint(allocatedSize * 1.5f), (currentSize * 2) - 1);


		CudaCheck(cudaMallocManaged((void**)&nodeTemp, allocatedSize * sizeof(Node)));

		CudaCheck(cudaFree(bvh));
		bvh = nodeTemp;
	}

	root = bvh;

	uint blockSize = 64;
	uint gridSize = (currentSize + blockSize - 1) / blockSize;

	CudaCheck(cudaDeviceSynchronize());

	Reset << <gridSize, blockSize >> >(currentSize, bvh, *data, *mortonCodes, currentSize - 1);
	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());

	BuildTree << <gridSize, blockSize >> >(currentSize, bvh, *data, *mortonCodes, currentSize - 1,this);

	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());

	CudaCheck(cudaEventRecord(stop, 0));
	CudaCheck(cudaEventSynchronize(stop));
	CudaCheck(cudaEventElapsedTime(&time, start, stop));
	CudaCheck(cudaEventDestroy(start));
	CudaCheck(cudaEventDestroy(stop));

	LOG(TRACE, "     BVH Creation Execution: " , time , "ms" );
}