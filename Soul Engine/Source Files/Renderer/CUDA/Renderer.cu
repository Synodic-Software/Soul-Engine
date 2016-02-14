#include "Renderer.cuh"

__global__ void IntegrateKernal(const uint n, RayJob* job, const uint counter){


	uint index = getGlobalIdx_1D_1D();

	if (index < n){

		((glm::vec4*)job->GetResultPointer(0))[index] = glm::mix(((glm::vec4*)job->GetResultPointer(0))[index], ((glm::vec4*)job->GetResultPointer(1))[index], 1.0f / counter);
	}
}


__host__ void Integrate(RayJob* RenderJob,const uint counter){
	RenderJob->SwapResults(0, 1);

	uint n = RenderJob->GetRayAmount();
	uint blockSize = 64;
	uint gridSize = (n + blockSize - 1) / blockSize;

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	IntegrateKernal << <gridSize, blockSize >> >(n,RenderJob,counter);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout << "RayClear Execution: " << time << "ms" << std::endl;
}