#pragma once
#include <glm/vec2.hpp>
#include "GPGPU/GPUBuffer.h"

namespace Filter {

	void HermiteBicubic(GPUBuffer<glm::vec4>& data, glm::uvec2 originalSize, glm::uvec2 desiredSize);

}
