#pragma once
#include "GPGPU/GPUBuffer.h"
#include "glm/glm.hpp"
#include "Metrics.h"
#include "Utility\CUDA\CUDAHelper.cuh"

class Film {
public:
	Film();
	~Film();

	/*
	 *    Gets 1 d.
	 *    @param	parameter1	The first parameter.
	 *    @return	The data at the specified index.
	 */

	__host__ __device__ uint GetIndex(uint);

	/*
	 *    Gets a normalized.
	 *    @param	parameter1	The first parameter.
	 *    @return	The normalized.
	 */

	__host__ __device__ glm::vec2 GetNormalized(uint);

	glm::uvec2 resolution;
	
private:

	//TODO share this map with other Film Object of different resolution
	uint64* indexMap;

	GPUBuffer* data;


};
