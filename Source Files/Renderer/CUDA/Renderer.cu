#include "Renderer.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"

__global__ void IntegrateKernal(const uint n, glm::vec4* A, glm::vec4* B, const uint counter){


	uint index = getGlobalIdx_1D_1D();

	if (index < n){
		B[index] = glm::mix(B[index], A[index], 1.0f / counter);
		A[index] = B[index];
	}
}


__host__ void Integrate(RayJob* RenderJob, glm::vec4* A, glm::vec4* B,const uint counter){
	//RenderJob->SwapResults(0, 1);

	uint n = RenderJob->rayAmount;
	uint blockSize = 64;
	uint gridSize = (n + blockSize - 1) / blockSize;

	cudaEvent_t start, stop;
	float time;
	CudaCheck(cudaEventCreate(&start));
	CudaCheck(cudaEventCreate(&stop));
	CudaCheck(cudaEventRecord(start, 0));

	IntegrateKernal << <gridSize, blockSize >> >(n, A,B, counter);
	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());
	CudaCheck(cudaEventRecord(stop, 0));
	CudaCheck(cudaEventSynchronize(stop));
	CudaCheck(cudaEventElapsedTime(&time, start, stop));
	CudaCheck(cudaEventDestroy(start));
	CudaCheck(cudaEventDestroy(stop));

	std::cout << "Colour Merge Execution: " << time << "ms" << std::endl;
}