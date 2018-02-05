#pragma once

#include "Compute/ComputeBuffer.h"
#include "CUDA/BVH.cuh"

class BVH {

public:

	BVH();

	void Build(int, ComputeBuffer<BVHData>&, ComputeBuffer<uint64>&, ComputeBuffer<BoundingBox>&);


private:

	ComputeBuffer<Node> nodes;

};
