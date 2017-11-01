#include "Filter.h"

#include "CUDA/Filter.cuh"

void Filter::IterativeBicubic(glm::vec4* data, glm::uvec2 originalSize, glm::uvec2 desiredSize) {
	CUDAFilter::HermiteBicubic(data,originalSize,desiredSize);
}