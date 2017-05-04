#pragma once

#include <cuda_runtime.h>

#include "Utility\Includes\GLMIncludes.h"
#include "Metrics.h"

class Face
{
public:
	__host__ __device__ Face();
	Face(glm::uvec3, uint);
	__host__ __device__ ~Face();

	glm::uvec3 indices;
	uint material;
	uint64 mortonCode;
	bool isCamera;
	__host__ __device__ bool operator==(const Face& other) const {
		return
			indices == other.indices &&
			material == other.material&&
			mortonCode == other.mortonCode;
	}

	__host__ __device__ friend void swap(Face& a, Face& b)
	{

		glm::uvec3 temp = a.indices;
		a.indices = b.indices;
		b.indices = temp;

		uint temp1 = a.material;
		a.material = b.material;
		b.material = temp1;

	}
	__host__ __device__ Face& operator=(Face arg)
	{
		this->indices = arg.indices;
		this->material = arg.material;
		this->mortonCode = arg.mortonCode;

		return *this;
	}
private:

};
