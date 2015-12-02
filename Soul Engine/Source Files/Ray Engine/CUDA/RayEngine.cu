#include "RayEngine.cuh"

uint raySeedGl=0;


inline CUDA_FUNCTION uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}


inline __device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}


inline __device__ void GetCurrentJob(RayJob *&job, const uint& index, uint& startIndex){

	while (job->nextRay != NULL && !(index < startIndex + job->rayAmount*job->samples)){
		startIndex += job->rayAmount*job->samples;
		job = job->nextRay;
	}

}

__global__ void EngineResultClear(const uint n, RayJob* job){


	uint index = getGlobalIdx_1D_1D();

	if (index < n){

		uint startIndex = 0;

		GetCurrentJob(job, index, startIndex);

		job->resultsT[(index - startIndex) / job->samples] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}
}

__global__ void EngineExecute(const uint n, RayJob* job, const uint raySeed, const Scene* scene){

	uint index = getGlobalIdx_1D_1D();

	if (index < n){

		thrust::default_random_engine rng(randHash(raySeed) * randHash(index));
		thrust::uniform_real_distribution<float> uniformDistribution(0.0f, 1.0f);


		uint index = getGlobalIdx_1D_1D();
		uint startIndex = 0;

		GetCurrentJob(job, index, startIndex);

		uint localIndex = index - startIndex / job->samples;

		Ray ray;
		job->camera->SetupRay(localIndex, ray, rng, uniformDistribution);


		//calculate something
		glm::vec3 col =scene->IntersectColour(ray);

		
		atomicAdd(&(job->resultsT[localIndex].x), col.x / job->samples);

		atomicAdd(&(job->resultsT[localIndex].y), col.y / job->samples);

		atomicAdd(&(job->resultsT[localIndex].z), col.z / job->samples);

	}
}


__host__ void ProcessJobs(RayJob* jobs, const Scene* scene){
	raySeedGl++;
	CudaCheck(cudaDeviceSynchronize());
	if (jobs!=NULL){
	uint n = 0;

	RayJob* temp = jobs;
	n += temp->rayAmount*temp->samples;
	while (temp->nextRay != NULL){
		temp = temp->nextRay;
		n += temp->rayAmount*temp->samples;
	}

	if (n!=0){

		uint blockSize = 32;
		uint gridSize = (n + blockSize - 1) / blockSize;


		//execute engine


		cudaEvent_t start, stop; 
		float time;
		cudaEventCreate(&start); 
		cudaEventCreate(&stop); 
		cudaEventRecord(start, 0);


		EngineResultClear << <gridSize, blockSize >> >(n, jobs);

		EngineExecute << <gridSize, blockSize >> >(n, jobs, WangHash(raySeedGl), scene);

		cudaEventRecord(stop, 0); 
		cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&time, start, stop); 
		cudaEventDestroy(start); 
		cudaEventDestroy(stop);

		std::cout << "RayEngine Execution: " << time << "ms"<< std::endl;

		CudaCheck(cudaDeviceSynchronize());
	}
	}


}