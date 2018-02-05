#pragma once

#include "Data Structures\Bounding Volume Heirarchy\Node.h"

typedef struct BVHData {

	uint root;
	uint leafSize;
	Node* nodes;

	__device__ bool IsLeaf(uint test) const{
		return test >= leafSize;
	}

}BVHData;

__global__ void BuildTree(uint, BVHData*, Node*, uint64*, BoundingBox*);
__global__ void Reset(uint, Node*);