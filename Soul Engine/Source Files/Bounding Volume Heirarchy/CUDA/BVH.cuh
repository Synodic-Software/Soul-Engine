#pragma once

#include "Utility\CUDAIncludes.h"
#include "Bounding Volume Heirarchy\CUDA\Node.cuh"
#include "Bounding Volume Heirarchy\BoundingBox.h"
#include "Node.cuh"
#include "Engine Core\Object\Face.h"

class BVH : public Managed{
public:

	BVH(Face*** datan, uint64** mortonCodesn);

	CUDA_FUNCTION Node* GetRoot(){
		return bvh;
	}
	CUDA_FUNCTION bool IsLeaf(Node* test){
		return ((test - bvh)>=(currentSize - 1));
	}
	CUDA_FUNCTION uint GetSize(){
		return currentSize;
	}
	void Build(uint);
private:
	Node* bvh;
	uint currentSize;
	uint allocatedSize;
	Face*** data;
	uint64** mortonCodes;


};
