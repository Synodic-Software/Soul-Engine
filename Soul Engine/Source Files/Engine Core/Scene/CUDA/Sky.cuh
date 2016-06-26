#pragma once

#include "Utility\CUDAIncludes.h"
#include "Engine Core/Material/Texture/Image.cuh"

class Sky : public Managed
{
public:
	Sky(std::string); 

	__host__ void UpdateSky();

	__device__ glm::vec3 ExtractColour(const glm::vec3& direction);

	Image* image;
private:


};