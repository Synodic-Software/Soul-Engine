#pragma once

#include "Utility\CUDAIncludes.h"
//__align__(32)
class  Face : public Managed
{
public:
	Face();
	Face(glm::uvec3, uint);
	~Face();

	void SetData(glm::uvec3, uint);

	glm::uvec3 indices;
	uint materialID;
	uint64 mortonCode;

private:
	
};
