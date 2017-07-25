#pragma once

#include "glm/glm.hpp"
#include "Metrics.h"

#include <curand_kernel.h>

class Film {
public:
	Film();
	~Film();

	/*
	 *    Gets 1 d.
	 *    @param	parameter1	The first parameter.
	 *    @return	The data at the specified index.
	 */

	__device__ uint GetIndex(uint);

	/*
	 *    Gets a normalized.
	 *    @param	parameter1	The first parameter.
	 *    @return	The normalized.
	 */

	__device__ glm::vec2 GetSample(uint, curandState&);

	glm::uvec2 resolution;
	uint* indicePointer;


	bool operator==(const Film& other) const {
		return
			resolution == other.resolution &&
			indicePointer == other.indicePointer;
	}

	Film& operator=(Film arg)
	{
		this->resolution = arg.resolution;
		this->indicePointer = arg.indicePointer;

		return *this;
	}
private:

};
