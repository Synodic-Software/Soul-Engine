#pragma once

#include "Compute/ComputeBuffer.h"
#include "Node.h"
#include "Engine Core/Object/Face.h"
#include "Engine Core/Object/Vertex.h"
#include "CUDA/BVH.cuh"

class BVH {
public:

	BVH();
	~BVH() = default;

	void Build(int, ComputeBuffer<BVHData>&, ComputeBuffer<uint64>&, ComputeBuffer<Face>&, ComputeBuffer<Vertex>&);

private:
	ComputeBuffer<Node> bvh;

};