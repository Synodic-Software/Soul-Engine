#pragma once

#include "Metrics.h"
#include "Utility\CUDA\CUDAHelper.cuh"

#include <glm/vec2.hpp>
#include "GPGPU/GPUBuffer.h"

namespace CUDAFilter {

	__host__ void HermiteBicubic(GPUBuffer<glm::vec4>& data, glm::uvec2 originalSize, glm::uvec2 desiredSize);
	
}
