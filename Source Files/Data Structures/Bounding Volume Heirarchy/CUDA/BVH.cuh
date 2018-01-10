#pragma once

#include "Data Structures\Bounding Volume Heirarchy\CUDA\Node.cuh"
#include "Data Structures\Bounding Volume Heirarchy\BoundingBox.h"
#include "Engine Core\Object\Face.h"
#include "Engine Core\Object\Vertex.h"
#include "GPGPU/ComputeBuffer.h"

typedef struct BVHData {

	Node* root;
	uint currentSize;
	Node* bvh;

	__device__ BVHData& operator=(const BVHData& arg)
	{
		this->root = arg.root;
		this->currentSize = arg.currentSize;

		return *this;
	}

	__inline__ __device__ bool IsLeaf(Node* test) {
		return ((test - bvh) >= (currentSize - 1) && (test - bvh) < currentSize * 2 - 1);
	}

	__inline__ __device__ Node* GetLeaf(int test) {
		return bvh + ((currentSize - 1) + test);
	}

}BVHData;

__global__ void BuildTree(uint n, BVHData* data, Node* nodes, uint64* mortonCodes, uint leafOffset);
__global__ void Reset(uint n, Node* nodes, Face* faces, Vertex* vertices, uint64* mortonCodes, uint leafOffset);