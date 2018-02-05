#pragma once

#include "Data Structures\Bounding Volume Heirarchy\InnerNode.h"
#include "Data Structures\Bounding Volume Heirarchy\LeafNode.h"


typedef struct BVHData {

	uint root;
	uint leafSize;
	InnerNode* innerNodes;
	LeafNode* leafNodes;

}BVHData;

__global__ void BuildTree(uint, BVHData*, InnerNode*, LeafNode*, uint64*, BoundingBox*);
__global__ void Reset(uint, InnerNode*, LeafNode*);