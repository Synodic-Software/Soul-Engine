#pragma once

#include <cuda_runtime.h>

#include "Utility\Includes\GLMIncludes.h"
#include "Metrics.h"

class Tet
{
public:
	__host__ __device__ Tet();
	Tet(glm::uvec4, uint);
	__host__ __device__ ~Tet();


	glm::uvec4 indices;
	uint material;
	uint object;

	__host__ __device__ bool operator==(const Tet& other) const {
		return
			indices == other.indices &&
			material == other.material &&
			object == other.object;
	}

	__host__ __device__ friend void swap(Tet& a, Tet& b)
	{

		glm::uvec4 temp = a.indices;
		a.indices = b.indices;
		b.indices = temp;

		uint temp1 = a.material;
		a.material = b.material;
		b.material = temp1;

		temp1 = a.object;
		a.object = b.object;
		b.object = temp1;

	}
	__host__ __device__ Tet& operator=(Tet arg)
	{
		this->indices = arg.indices;
		this->material = arg.material;
		this->object = arg.object;

		return *this;
	}

private:
	
};
