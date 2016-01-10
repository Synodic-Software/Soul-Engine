#pragma once

#include "Utility\CUDAIncludes.h"

class Face : public Managed
{
public:
	Face();
	Face(glm::uvec3, uint);
	~Face();

	void SetData(glm::uvec3, uint);

	glm::uvec3 indices;
	uint materialID;
private:
	
};
