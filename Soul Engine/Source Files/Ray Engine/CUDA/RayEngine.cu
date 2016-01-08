#include "RayEngine.cuh"

uint raySeedGl=0;

thrust::host_vector<RayJob*> hostJobList(0);
thrust::device_vector<RayJob*> deviceJobList(0);

// Template structure to pass to kernel
template <typename T>
struct KernelArray : public Managed
{
private:
	T*  _array;
	int _size;

public:
	__device__ int size(){
		return _size;
	}

	__device__ T operator[](int i) {
		return _array[i]; 
	}

	// constructor allows for implicit conversion
	__host__ KernelArray(thrust::device_vector<T>& dVec) {
		_array = thrust::raw_pointer_cast(&dVec[0]);
		_size = (int)dVec.size();
	}

};

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


inline __device__ int GetCurrentJob(KernelArray<RayJob*>& jobList, const uint& index, uint& startIndex){

	int i = 0;
	for (; 
		i<jobList.size() && !(index < startIndex + jobList[i]->GetRayAmount()*jobList[i]->GetSampleAmount());
		i++){

	}
	return i;
	//while (job->nextRay != NULL && !(index < startIndex + job->rayAmount*job->samples)){
	//	startIndex += job->rayAmount*job->samples;
	//	job = job->nextRay;
	//}

}

__global__ void EngineResultClear(const uint n, KernelArray<RayJob*> job){


	uint index = getGlobalIdx_1D_1D();

	if (index < n){

		uint startIndex = 0;

		int cur=GetCurrentJob(job, index, startIndex);

		
		((glm::vec4*)job[cur]->GetResultPointer())[(index - startIndex) / int(glm::ceil(job[cur]->GetSampleAmount()))] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}
}

__global__ void EngineExecute(const uint n, KernelArray<RayJob*> job, const uint raySeed, const Scene* scene){

	uint index = getGlobalIdx_1D_1D();

	uint startIndex = 0;
	int cur = GetCurrentJob(job, index, startIndex);


	thrust::default_random_engine rng(randHash(raySeed) * randHash(index));
	thrust::uniform_real_distribution<float> uniformDistribution(0.0f, 1.0f);

	float prob = uniformDistribution(rng);

	if (index < n){
		
		uint localIndex = index - startIndex / glm::ceil(job[cur]->GetSampleAmount());


			Ray ray;
			job[cur]->GetCamera()->SetupRay(localIndex, ray, rng, uniformDistribution);


			//calculate something
			glm::vec3 col = scene->IntersectColour(ray);



			glm::vec4* pt = ((glm::vec4*)job[cur]->GetResultPointer());

			atomicAdd(&pt[localIndex].x, col.x / glm::ceil(job[cur]->GetSampleAmount()));

			atomicAdd(&pt[localIndex].y, col.y / glm::ceil(job[cur]->GetSampleAmount()));

			atomicAdd(&pt[localIndex].z, col.z / glm::ceil(job[cur]->GetSampleAmount()));

	}
}


__host__ void ProcessJobs(std::vector<RayJob*>& jobs, const Scene* scene){
	raySeedGl++;
	CudaCheck(cudaDeviceSynchronize());
	if (jobs.size()>0){

	uint n = 0;
	hostJobList.clear();
	for (int i = 0; i < jobs.size();i++ ){
		n += jobs[i]->GetRayAmount()*jobs[i]->GetSampleAmount();
		hostJobList.push_back(jobs[i]);
	}

	if (n!=0){

		deviceJobList = hostJobList;

		uint blockSize = 32;
		uint gridSize = (n + blockSize - 1) / blockSize;


		//execute engine


		cudaEvent_t start, stop; 
		float time;
		cudaEventCreate(&start); 
		cudaEventCreate(&stop); 
		cudaEventRecord(start, 0);


		KernelArray<RayJob*> jobL = KernelArray<RayJob*>(deviceJobList);

		EngineResultClear << <gridSize, blockSize >> >(n, jobL);

		EngineExecute << <gridSize, blockSize >> >(n, jobL, WangHash(raySeedGl), scene);

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