#pragma once

#include "Utility\CUDA\CUDAHelper.cuh"

#include <glm/glm.hpp>

namespace CUDAFilter {

	__host__ void HermiteBicubic(glm::vec4* data, glm::uvec2& originalSize, glm::uvec2& desiredSize);
	
}
