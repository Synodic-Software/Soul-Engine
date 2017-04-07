#pragma once

#include "Engine Core/Material/Texture/Image.cuh"
#include <string>
#include "Utility\Includes\GLMIncludes.h"

	



class Sky
{

public:

	Sky(std::string); 

	__host__ void UpdateSky();

	__device__ glm::vec3 ExtractColour(const glm::vec3& direction);

	Image* image;

private:


	
};


