#pragma once

#include "Compute/ComputeBuffer.h"
#include "Data Structures/Geometric Primatives/Face.h"
#include "Data Structures/Geometric Primatives/Vertex.h"
#include "CUDA/BVH.cuh"

class BVH {

public:

	BVH();

	void Build(int, ComputeBuffer<BVHData>&, ComputeBuffer<uint64>&, ComputeBuffer<BoundingBox>&);


private:

	ComputeBuffer<InnerNode> innerNodes;
	ComputeBuffer<LeafNode> leafNodes;

};
