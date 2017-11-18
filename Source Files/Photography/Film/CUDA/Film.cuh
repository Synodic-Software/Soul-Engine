#pragma once

#include "glm/glm.hpp"
#include "Metrics.h"

#include <curand_kernel.h>
#include "GPGPU/GPUTexture.h"

class Film {
public:
	Film();
	~Film();

	/*
	 *    Gets a normalized.
	 *    @param	parameter1	The first parameter.
	 *    @return	The normalized.
	 */

	__device__ glm::vec2 GetSample(uint, curandState&);

	glm::uvec2 resolutionPrev;
	glm::uvec2 resolution;
	glm::uvec2 resolutionMax;

	float resolutionRatio;

	//GPUTexture<glm::vec4> results;
	glm::vec4* results;
	int* hits;

	bool operator==(const Film& other) const {
		return
			resolutionPrev == other.resolutionPrev &&
			resolution == other.resolution &&
			resolutionMax == other.resolutionMax &&
			resolutionRatio == other.resolutionRatio &&
			results == other.results &&
			hits == other.hits;
			
	}

	Film& operator=(const Film& arg)
	{
		this->resolutionPrev = arg.resolutionPrev;
		this->resolution = arg.resolution;
		this->resolutionMax = arg.resolutionMax;
		this->resolutionRatio = arg.resolutionRatio;
		this->results = arg.results;
		this->hits = arg.hits;

		return *this;
	}
private:

};
