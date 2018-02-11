#pragma once

#include "Compute/ComputeBuffer.h"
#include "CUDA/BVH.cuh"

class LBVHManager {

public:

	LBVHManager();

	void Build(int, ComputeBuffer<BVH>&, ComputeBuffer<uint64>&, ComputeBuffer<BoundingBox>&);


private:

	ComputeBuffer<Node> nodes;

};
