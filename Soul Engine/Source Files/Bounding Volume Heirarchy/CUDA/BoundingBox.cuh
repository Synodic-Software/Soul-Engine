#pragma once

#include "Engine Core\BasicDependencies.h"

class BoundingBox : public Managed
{
public:
	CUDA_FUNCTION BoundingBox();
	CUDA_FUNCTION BoundingBox(glm::vec3, glm::vec3);

	CUDA_FUNCTION ~BoundingBox();

private: 
	glm::vec3 origin;
	glm::vec3 extent;
};

