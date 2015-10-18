#include "Ray Engine\RayEngine.cuh"

uint raySeedGl=0;


inline __device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

inline __device__ bool AABBIntersect(glm::vec3& origin, glm::vec3& extent, glm::vec3& o, glm::vec3& dInv, float t0, float t1){

	glm::vec3 boxMax = origin + extent;
	glm::vec3 boxMin = origin - extent;

	float tx1 = (boxMin.x - o.x)*dInv.x;
	float tx2 = (boxMax.x - o.x)*dInv.x;

	float tmin = glm::min(tx1, tx2);
	float tmax = glm::max(tx1, tx2);

	float ty1 = (boxMin.y - o.y)*dInv.y;
	float ty2 = (boxMax.y - o.y)*dInv.y;

	tmin = glm::max(tmin, glm::min(ty1, ty2));
	tmax = glm::min(tmax, glm::max(ty1, ty2));

	float tz1 = (boxMin.z - o.z)*dInv.z;
	float tz2 = (boxMax.z - o.z)*dInv.z;

	tmin = glm::max(tmin, glm::min(tz1, tz2));
	tmax = glm::min(tmax, glm::max(tz1, tz2));

	return tmax >= glm::max(t0, tmin) && tmin < t1;

}

inline __device__ float4 Intersect(Ray& ray){



	if (AABBIntersect(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), ray.origin, 1.0f / ray.direction, 0.0f, 4294967295.0f)){
		return make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	}
	else{
		return make_float4(0.0f, 0.0f, 0.0f,1.0f);
	}
}
__global__ void EngineResultClear(uint n, RayJob* jobs){
	uint index = getGlobalIdx_1D_1D();

	RayJob* job = jobs;
	uint startIndex = 0;

	while (jobs->nextRay != NULL && !(index < startIndex + job->rayAmount*job->samples)){
		startIndex += job->rayAmount*job->samples;
		job = job->nextRay;
	}

	uint localJob = index - startIndex;

	uint localIndex = localJob / job->samples;

	job->resultsT[localIndex] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}
__global__ void EngineExecute(uint n, RayJob* jobs, uint raySeed){

	uint index = getGlobalIdx_1D_1D();

	if (index < n){

	thrust::default_random_engine rng(randHash(raySeed) * randHash(index));
	thrust::uniform_real_distribution<float> uniformDistribution(0.0f, 1.0f);


		RayJob* job = jobs;
		uint startIndex = 0;

		while (jobs->nextRay != NULL && !(index < startIndex + job->rayAmount*job->samples)){
			startIndex += job->rayAmount*job->samples;
			job = job->nextRay;
		}

		uint localJob = index - startIndex;

		uint localIndex = localJob / job->samples;

		Ray ray;
		job->camera->SetupRay(localIndex, ray, rng, uniformDistribution);

		//uint x = localIndex / job->camera->resolution.x;
		//uint y = localIndex % job->camera->resolution.y;

		//calculate something


			
		float4 col = Intersect(ray);
		float4 accum = job->resultsT[localIndex];
		job->resultsT[localIndex] = make_float4(accum.x + col.x, accum.y + col.y, accum.z + col.z, 1.0f);

	}
}

__host__ void ProcessJobs(RayJob* jobs){
	raySeedGl++;

	if (jobs!=NULL){
	uint n = 0;

	RayJob* temp = jobs;
	n += temp->rayAmount*temp->samples;
	while (temp->nextRay != NULL){
		temp = temp->nextRay;
		n += temp->rayAmount*temp->samples;
	}

	if (n!=0){

		int blockSize;   // The launch configurator returned block size 
		int minGridSize; // The minimum grid size needed to achieve the 
		// maximum occupancy for a full device launch 
		int gridSize;    // The actual grid size needed, based on input size 

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
			EngineExecute, 0, 0);
		// Round up according to array size 
		gridSize = (n + blockSize - 1) / blockSize;


		//execute engine


		cudaEvent_t start, stop; 
		float time; 
		cudaEventCreate(&start); 
		cudaEventCreate(&stop); 
		cudaEventRecord(start, 0);

		EngineResultClear << <gridSize, blockSize >> >(n, jobs);
		EngineExecute << <gridSize, blockSize >> >(n, jobs, raySeedGl);

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