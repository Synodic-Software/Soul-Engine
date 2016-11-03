#include "Renderer.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"

__global__ void IntegrateKernal(const uint n, RayJob* job, const uint counter){


	uint index = getGlobalIdx_1D_1D();

	if (index < n){
		((glm::vec4*)job->GetResultPointer(1))[index] = glm::mix(((glm::vec4*)job->GetResultPointer(1))[index], ((glm::vec4*)job->GetResultPointer(0))[index], 1.0f / counter);
		((glm::vec4*)job->GetResultPointer(0))[index] = ((glm::vec4*)job->GetResultPointer(1))[index];
	}
}


__host__ void Integrate(RayJob* RenderJob,const uint counter){
	//RenderJob->SwapResults(0, 1);

	uint n = RenderJob->GetRayAmount();
	uint blockSize = 64;
	uint gridSize = (n + blockSize - 1) / blockSize;

	cudaEvent_t start, stop;
	float time;
	CudaCheck(cudaEventCreate(&start));
	CudaCheck(cudaEventCreate(&stop));
	CudaCheck(cudaEventRecord(start, 0));

	IntegrateKernal << <gridSize, blockSize >> >(n, RenderJob, counter);
	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());
	CudaCheck(cudaEventRecord(stop, 0));
	CudaCheck(cudaEventSynchronize(stop));
	CudaCheck(cudaEventElapsedTime(&time, start, stop));
	CudaCheck(cudaEventDestroy(start));
	CudaCheck(cudaEventDestroy(stop));

	std::cout << "Colour Merge Execution: " << time << "ms" << std::endl;
}