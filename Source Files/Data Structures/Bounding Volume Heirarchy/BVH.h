#pragma once

#include "GPGPU/GPUBuffer.h"
#include "Node.h"
#include "Engine Core/Object/Face.h"
#include "Engine Core/Object/Vertex.h"
#include "CUDA/BVH.cuh"

class BVH {
public:

	BVH();
	~BVH();

	void Build(int, GPUBuffer<BVHData>&, GPUBuffer<uint64>&, GPUBuffer<Face>&, GPUBuffer<Vertex>&);

private:
	GPUBuffer<Node> bvh;

};
