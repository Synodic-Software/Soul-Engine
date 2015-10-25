#include "Ray Engine\RayEngine.cuh"

uint raySeedGl=0;


inline __device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}


inline __device__ glm::vec3 positionAlongRay(const Ray& ray, const float& t) {
	return ray.origin + t * ray.direction;
}
inline __device__ glm::vec3 computeBackgroundColor(const glm::vec3& direction) {
	float position = (dot(direction, normalize(glm::vec3(-0.5, 0.5, -1.0))) + 1) / 2;
	glm::vec3 interpolatedColor = (1.0f - position) * glm::vec3(0.5f, 0.5f, 1.0f) +position * glm::vec3(1.0f, 1.0f, 1.0f);
	return interpolatedColor * 1.0f;
}

inline __device__ bool FindTriangleIntersect(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
	const glm::vec3& o, const glm::vec3& d,
	float& lambda, float& bary1, float& bary2)
{
	glm::vec3 edge1 = b - a;
	glm::vec3 edge2 = c - a;

	glm::vec3 pvec = cross(d, edge2);
	float det = dot(edge1, pvec);
	if (det == 0.0f){
		return false;
	}
	float inv_det = 1.0f / det;

	glm::vec3 tvec = o - a;
	bary1 = dot(tvec, pvec) * inv_det;

	glm::vec3 qvec = cross(tvec, edge1);
	bary2 = dot(d, qvec) * inv_det;
	lambda = dot(edge2, qvec) * inv_det;

	bool hit = (bary1 >= 0.0f && bary2 >= 0.0f && (bary1 + bary2) <= 1.0f);
	return hit;
}

inline __device__ bool AABBIntersect(const glm::vec3& origin, const glm::vec3& extent, const glm::vec3& o, const glm::vec3& dInv, const float& t0, const float& t1){

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

inline __device__ float4 Intersect(const Ray& ray, RayJob *&job){



	if (AABBIntersect(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f), ray.origin, 1.0f / ray.direction, 0.0f, 4294967295.0f)){
		return make_float4(1.0f , 1.0f , 1.0f , 1.0f);
	}
	else{
		glm::vec3 backColor=computeBackgroundColor(ray.direction);
		return make_float4(backColor.x , backColor.y , backColor.z , 1.0f);
	}
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

__global__ void EngineExecute(const uint n, RayJob* job, const uint raySeed){

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

		uint x = localIndex / job->camera->resolution.x;
		uint y = localIndex % job->camera->resolution.y;

		//calculate something



		float4 col = Intersect(ray, job);
		
		atomicAdd(&(job->resultsT[localIndex].x), col.x / job->samples);

		atomicAdd(&(job->resultsT[localIndex].y), col.y / job->samples);

		atomicAdd(&(job->resultsT[localIndex].z), col.z / job->samples);

	}
}


__host__ void ProcessJobs(RayJob* jobs){
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

		EngineExecute     << <gridSize, blockSize >> >(n, jobs, raySeedGl);

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