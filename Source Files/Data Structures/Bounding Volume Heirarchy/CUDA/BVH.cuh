#pragma once

#include "Data Structures\Bounding Volume Heirarchy\InnerNode.h"
#include "Data Structures\Bounding Volume Heirarchy\LeafNode.h"

#include "Data Structures\Geometric Primatives\Face.h"
#include "Data Structures\Geometric Primatives\Vertex.h"
#include "Compute/ComputeBuffer.h"

typedef struct BVHData {

	uint root;
	uint leafSize;
	InnerNode* innerNodes;
	LeafNode* leafNodes;

	//__inline__ __device__ bool IsLeaf(Node* test) {
	//	return ((test - bvh) >= (leafSize - 1) && (test - bvh) < leafSize * 2 - 1);
	//}

	//__inline__ __device__ Node* GetLeaf(int test) {
	//	return bvh + ((currentSize - 1) + test);
	//}

}BVHData;

__global__ void BuildTree(uint, uint,BVHData*, InnerNode*, LeafNode*, uint64*);
__global__ void Reset(uint, uint, InnerNode*, LeafNode*, Face*, Vertex*);