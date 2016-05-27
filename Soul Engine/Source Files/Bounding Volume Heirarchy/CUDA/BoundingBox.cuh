#pragma once

#include "Utility/CUDAIncludes.h"

class __align__(32) BoundingBox : public Managed
{
public:
	CUDA_FUNCTION BoundingBox();
	CUDA_FUNCTION BoundingBox(glm::vec3, glm::vec3);

	CUDA_FUNCTION BoundingBox& BoundingBox::operator= (const BoundingBox &a)
	{

		min = a.min;
		max = a.max;

		return *this;
	}

	CUDA_FUNCTION ~BoundingBox();

	glm::vec3 min;
	glm::vec3 max;
private: 
	
};