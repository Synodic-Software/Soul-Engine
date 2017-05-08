#pragma once

#include "Metrics.h"
#include "Bounding Volume Heirarchy\BVH.h"

#include "Engine Core\Object\Face.h"
#include "Engine Core\Object\Vertex.h"

namespace MortonCode{


	__global__ void ComputeGPU64(const uint n, uint64* mortonCodes, Face* faces, Vertex* vertices);

	//given a point in space with the range [-1,1] for each dimension
	__host__ __device__ uint64 Calculate64(const glm::vec3&);

	//returns a point in space with the range [-1,1] for each dimension 
	__host__ __device__ glm::vec3 Decode64(const uint64);

}