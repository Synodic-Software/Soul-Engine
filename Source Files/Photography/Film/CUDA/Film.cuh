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


	bool operator==(const Film& other) const {
		return
			resolution == other.resolution;
	}

	Film& operator=(Film arg)
	{
		this->resolution = arg.resolution;

		return *this;
	}
private:

};
