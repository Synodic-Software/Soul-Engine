#pragma once

#include "Metrics.h"

//#include <curand_kernel.h>
#include "Compute/GPUTexture.h"

//#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

class Film {
public:
	Film();
	~Film();

	/*
	 *    Gets a normalized.
	 *    @param	parameter1	The first parameter.
	 *    @return	The normalized.
	 */

	//__device__ glm::vec2 GetSample(uint, curandState&);

	glm::uvec2 resolutionPrev;
	glm::uvec2 resolution;
	glm::uvec2 resolutionMax;

	float resolutionRatio;

	//GPUTexture<glm::vec4> results;
	glm::vec4* results;
	int* hits;

private:

};
