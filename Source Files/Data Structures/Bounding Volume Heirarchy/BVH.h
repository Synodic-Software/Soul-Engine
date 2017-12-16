#pragma once

#include "CUDA\Node.cuh"
#include "BoundingBox.h"
#include "Engine Core\Object\Face.h"
#include "Engine Core\Object\Vertex.h"

typedef struct BVHData {

	Node* root;
	uint currentSize;
	Node* bvh;

	__device__ BVHData& operator=(BVHData& arg)
	{
		this->root = arg.root;
		this->currentSize = arg.currentSize;

		return *this;
	}

	__inline__ __device__ bool IsLeaf(Node* test) const {
		return ((test - bvh) >= (currentSize - 1) && (test - bvh) < currentSize * 2 - 1);
	}

	__inline__ __device__ Node* GetLeaf(const int test) const {
		return bvh + ((currentSize - 1) + test);
	}

}BVHData;

class BVH {
public:

	__host__ BVH();
	__host__ ~BVH();

	__host__ void Build(uint, BVHData*&, uint64*, Face *, Vertex*);

private:
	BVHData bvhDataHost;
	Node* bvh;
	uint allocatedSize;

};
