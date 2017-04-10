#pragma once

#include "Bounding Volume Heirarchy\CUDA\Node.cuh"
#include "Bounding Volume Heirarchy\BoundingBox.h"
#include "Engine Core\Object\Face.h"

class BVH{
public:

	BVH(Face*** datan, uint64** mortonCodesn);

	__device__ Node* GetRoot(){
		return root;
	}
	__device__ Node* GetNodes(){
		return bvh;
	}
	__device__ bool IsLeaf(Node* test){
		return ((test - bvh) >= (currentSize - 1));
	}
	__device__ uint GetSize(){
		return currentSize;
	}
	__device__ Node* GetLeaf(int test){
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
