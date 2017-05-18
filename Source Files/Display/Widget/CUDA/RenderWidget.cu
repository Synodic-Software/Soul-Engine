#include "RenderWidget.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"

static uint allocatedSize = 0;
static uint* weights = 0;

__global__ void IntegrateKernal(const uint n, glm::vec4* A, glm::vec4* B, int* mask, uint* counters, const uint counter) {


	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	if (mask[index] > 0) {
		counters[index]++;

		B[index] = glm::mix(B[index], A[index], 1.0f / counters[index]);
	}

	A[index] = B[index];

}


__host__ void Integrate(uint size, glm::vec4* A, glm::vec4* B, int* mask, const uint counter) {

	if (size > allocatedSize) {
		allocatedSize = size;
		CudaCheck(cudaMalloc((void**)&weights, size * sizeof(uint)));
	}

	//clear b and 
	if (counter == 1) {
		cudaMemset(weights, 0, size * sizeof(uint));
		cudaMemset(B, 0, size * sizeof(glm::vec4));
	}

	uint blockSize = 64;
	uint gridSize = (size + blockSize - 1) / blockSize;

	IntegrateKernal << <gridSize, blockSize >> > (size, A, B, mask, weights, counter);
	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());

}