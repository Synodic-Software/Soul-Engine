#include "Filter.h"

#include "CUDA/Filter.cuh"

void Filter::HermiteBicubic(GPUBuffer<glm::vec4>& data, glm::uvec2 originalSize, glm::uvec2 desiredSize) {
	CUDAFilter::HermiteBicubic(data,originalSize,desiredSize);
}