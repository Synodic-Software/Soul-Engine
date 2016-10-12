#pragma once

#include "Bounding Volume Heirarchy\CUDA\Node.cuh"
#include "Bounding Volume Heirarchy\BoundingBox.h"
#include "Engine Core\Object\Face.h"

class BVH : public Managed{
public:

	BVH(Face*** datan, uint64** mortonCodesn);

	__host__ __device__ Node* GetRoot(){
		return root;
	}
	__host__ __device__ Node* GetNodes(){
		return bvh;
	}
	__host__ __device__ bool IsLeaf(Node* test){
		return ((test - bvh) >= (currentSize - 1));
	}
	__host__ __device__ uint GetSize(){
		return currentSize;
	}
	__host__ __device__ Node* GetLeaf(int test){
		return bvh+((currentSize - 1) + test);
	}
	void Build(uint);
	Node* root;
private:

	
	Node* bvh;
	uint currentSize;
	uint allocatedSize;
	Face*** data;
	uint64** mortonCodes;


};
