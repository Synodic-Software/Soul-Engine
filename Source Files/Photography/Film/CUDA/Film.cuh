#pragma once

#include "glm/glm.hpp"
#include "Metrics.h"

#include <curand_kernel.h>

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

	glm::uvec2 resolution;
	glm::uvec2 resolutionMax;
	float resolutionRatio;

	bool operator==(const Film& other) const {
		return
			resolution == other.resolution &&
			resolutionMax == other.resolutionMax &&
			resolutionRatio == other.resolutionRatio;
			
	}

	Film& operator=(const Film& arg)
	{
		this->resolution = arg.resolution;
		this->resolutionMax = arg.resolutionMax;
		this->resolutionRatio = arg.resolutionRatio;
		return *this;
	}
private:

};
