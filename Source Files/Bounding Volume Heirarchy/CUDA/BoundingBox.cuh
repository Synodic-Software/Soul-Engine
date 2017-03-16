#pragma once

#include "Utility\CUDA\CudaManaged.cuh"
#include <cuda_runtime.h>
#include "Utility\Includes\GLMIncludes.h"

class BoundingBox : public Managed
{
public:
	__host__ __device__ BoundingBox();
	__host__ __device__ BoundingBox(glm::vec3, glm::vec3);

	__host__ __device__ BoundingBox& BoundingBox::operator= (const BoundingBox &a)
	{

		min = a.min;
		max = a.max;

		return *this;
	}

	__host__ __device__ ~BoundingBox();

	glm::vec3 min;
	glm::vec3 max;
private: 
	
};