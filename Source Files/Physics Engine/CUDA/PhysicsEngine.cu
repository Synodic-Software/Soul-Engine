#include "PhysicsEngine.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"

typedef struct {

	Node* A;
	Node* B;

}Collision;

uint sizeAllocated = 0;
Collision* collisions = nullptr;


__device__ bool testAABBAABB(const BoundingBox& a, const BoundingBox& b)
{
	return(a.max.x > b.min.x &&
		a.min.x < b.max.x &&
		a.max.y > b.min.y &&
		a.min.y < b.max.y &&
		a.max.z > b.min.z &&
		a.min.z < b.max.z);
};


__global__ void NarrowPhase(uint n, const Scene* scene, Collision* collisions, int* sizeCounter){

	uint index = getGlobalIdx_1D_1D();

	if (index >= n){
		return;
	}





}

__global__ void BroadPhase(uint n, const Scene* scene, Collision* collisions, int* sizeCounter){

	uint index = getGlobalIdx_1D_1D();

	if (index >= n){
		return;
	}

	

	BVHData* bvh = scene->bvhData;
	
	Node* test = bvh->GetLeaf(index);

	Node* stack[64];
	short stackPtr = 0;

	stack[stackPtr++] = nullptr; // push

	// Traverse nodes starting from the root.
	Node* node = bvh->root;
	do
	{
		// Check each child node for overlap.
		Node* childL = node->childLeft;
		Node* childR = node->childRight;
		bool overlapL = (testAABBAABB(test->box,childL->box)
			&& childL != test);
		bool overlapR = (testAABBAABB(test->box,childR->box)
			&& childR != test);

		if (childL->rangeRight <= test - bvh->root)
			overlapL = false;

		if (childR->rangeRight <= test - bvh->root)
			overlapR = false;


		// Query overlaps a leaf node => report collision.
		if (overlapL && bvh->IsLeaf(childL))
			collisions[FastAtomicAdd(sizeCounter)] = { test, childL };

		if (overlapR && bvh->IsLeaf(childR))
			collisions[FastAtomicAdd(sizeCounter)] = { test, childR };

		// Query overlaps an internal node => traverse.
		bool traverseL = (overlapL && !bvh->IsLeaf(childL));
		bool traverseR = (overlapR && !bvh->IsLeaf(childR));

		if (!traverseL && !traverseR)
			node = stack[--stackPtr]; // pop
		else
		{
			node = (traverseL) ? childL : childR;
			if (traverseL && traverseR)
				stack[stackPtr++] = childR; // push
		}

	} while (node != nullptr);

}

__host__ void ProcessScene(const Scene* scene){
	CudaCheck(cudaDeviceSynchronize());

	uint n = scene->bvhData->currentSize;

	if (n <= 0){
		return;
	}

	uint maxEst = n*2.5f;

	if (maxEst > sizeAllocated){
		//deviceRays.resize(n);

		CudaCheck(cudaFree(collisions));

		CudaCheck(cudaMallocManaged((void**)&collisions, maxEst*sizeof(Collision)));
		sizeAllocated = maxEst;
	}


	int* sizeCounter;
	CudaCheck(cudaMallocManaged((void**)&sizeCounter, sizeof(int)));
	sizeCounter[0] = 0;

	CudaCheck(cudaDeviceSynchronize());


	int blockSize = 64;
	int gridSize = (n + blockSize - 1) / blockSize;
	BroadPhase << <gridSize, blockSize >> >(n, scene, collisions, sizeCounter);

	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());

	n = sizeCounter[0];
	sizeCounter[0] = 0;

	CudaCheck(cudaDeviceSynchronize());

	blockSize = 64;
	gridSize = (n + blockSize - 1) / blockSize;
	NarrowPhase << <gridSize, blockSize >> >(n, scene, collisions, sizeCounter);




	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());

	CudaCheck(cudaFree(sizeCounter));

}