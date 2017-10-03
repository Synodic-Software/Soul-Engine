#include "Filter.cuh"

__global__ void GPUHermiteBicubic(glm::vec4* data, glm::uvec2 originalSize, glm::uvec2 desiredSize) {

}

__host__ void CUDAFilter::HermiteBicubic(GPUBuffer<glm::vec4>& data, glm::uvec2 originalSize, glm::uvec2 desiredSize) {

	uint count = desiredSize.x*desiredSize.y;

	uint blockSize = 64;
	uint blockCount = (count + blockSize - 1) / blockSize;

	GPUHermiteBicubic << < blockCount, blockSize >> > (data, originalSize, desiredSize);

}
