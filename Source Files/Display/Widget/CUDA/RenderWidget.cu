#include "RenderWidget.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"

__global__ void IntegrateKernal(const uint n, glm::vec4* A, glm::vec4* B, const uint counter) {


	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	B[index] = glm::mix(B[index], A[index], 1.0f / counter);
	A[index] = B[index];

}


__host__ void Integrate(uint size, glm::vec4* A, glm::vec4* B, const uint counter) {

	uint blockSize = 64;
	uint gridSize = (size + blockSize - 1) / blockSize;

	IntegrateKernal << <gridSize, blockSize >> > (size, A, B, counter);
	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());

}