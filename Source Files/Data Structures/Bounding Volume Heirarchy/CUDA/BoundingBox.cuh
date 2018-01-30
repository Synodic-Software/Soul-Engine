#pragma once
#include <cuda_runtime.h>
#include "Utility\Includes\GLMIncludes.h"

class BoundingBox
{
public:
	__host__ __device__ BoundingBox();
	__host__ __device__ BoundingBox(glm::vec3, glm::vec3);

	glm::vec3 min;
	glm::vec3 max;
	
};