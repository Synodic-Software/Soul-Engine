#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cuh"

class Face : public Managed
{
public:
	Face();
	Face(glm::uvec3, uint);
	~Face();

	glm::uvec3 indices;
	uint materialID;
private:
	
};
