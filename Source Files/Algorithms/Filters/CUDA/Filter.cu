#include "Filter.cuh"

__global__ void GPUHermiteBicubic(uint n, glm::vec4* data, glm::uvec2 originalSize, glm::uvec2 desiredSize) {
	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

}

namespace CUDAFilter {

	__host__ void HermiteBicubic(glm::vec4* data, glm::uvec2& originalSize, glm::uvec2& desiredSize) {

		uint count = desiredSize.x*desiredSize.y;

		uint blockSize = 64;
		uint blockCount = (count + blockSize - 1) / blockSize;

		GPUHermiteBicubic << < blockCount, blockSize >> > (count, data, originalSize, desiredSize);
		CudaCheck(cudaPeekAtLastError());
		CudaCheck(cudaDeviceSynchronize());
	}

}