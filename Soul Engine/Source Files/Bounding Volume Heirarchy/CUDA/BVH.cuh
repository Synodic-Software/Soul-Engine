#pragma once

#include "Utility\CUDAIncludes.h"
#include "Bounding Volume Heirarchy\CUDA\Node.cuh"
#include "Bounding Volume Heirarchy\BoundingBox.h"
#include "Engine Core\Object\Face.h"

class BVH : public Managed{
public:

	BVH(Face*** datan, uint64** mortonCodesn);

	CUDA_FUNCTION Node* GetRoot(){
		return root;
	}
	CUDA_FUNCTION Node* GetNodes(){
		return bvh;
	}
	CUDA_FUNCTION bool IsLeaf(Node* test){
		return ((test - bvh) >= (currentSize - 1));
	}
	CUDA_FUNCTION uint GetSize(){
		return currentSize;
	}
	CUDA_FUNCTION Node* GetLeaf(int test){
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
