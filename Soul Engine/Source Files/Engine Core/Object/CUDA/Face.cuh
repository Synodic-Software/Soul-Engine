#pragma once

#include "Utility\CUDAIncludes.h"

class __align__(32) Face : public Managed
{
public:
	Face();
	Face(glm::uvec3, uint);
	~Face();

	void SetData(glm::uvec3, uint);

	glm::uvec3 indices;
	uint materialID;
	uint objectID;
	uint64 mortonCode;

private:
	
};
