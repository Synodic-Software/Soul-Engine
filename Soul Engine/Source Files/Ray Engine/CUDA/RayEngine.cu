#include "RayEngine.cuh"


uint raySeedGl = 0;

// Template structure to pass to kernel
template <typename T>
struct KernelArray : public Managed
{
private:
	T*  _array;
	int _size;

public:
	__device__ int Size(){
		return _size;
	}

	__device__ T operator[](int i) {
		return _array[i]; 
	}

	__host__ KernelArray() {
		_array = NULL;
		_size = 0;

	}

	// constructor allows for implicit conversion

	__host__ KernelArray(thrust::device_vector<T>& dVec) {
		_array = thrust::raw_pointer_cast(&dVec[0]);
		_size = (int)dVec.size();
	}

	__host__ ~KernelArray(){

	}

};

KernelArray<RayJob*> jobL;

inline CUDA_FUNCTION uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}


inline __device__ int GetCurrentJob(KernelArray<RayJob*>& jobList, const uint& index, uint& startIndex){

	int i = 0;
	for (; 
		i<jobList.Size() && !(index < startIndex + jobList[i]->GetRayAmount()*int(glm::ceil(jobList[i]->GetSampleAmount())));
		i++){

	}
	return i;
	//while (job->nextRay != NULL && !(index < startIndex + job->rayAmount*job->samples)){
	//	startIndex += job->rayAmount*job->samples;
	//	job = job->nextRay;
	//}

}

__global__ void EngineResultClear(const uint n, KernelArray<RayJob*> jobs){


	uint index = getGlobalIdx_1D_1D();

	if (index < n){

		uint startIndex = 0;

		int cur=GetCurrentJob(jobs, index, startIndex);

		
		((glm::vec4*)jobs[cur]->GetResultPointer(0))[(index - startIndex) / int(glm::ceil(jobs[cur]->GetSampleAmount()))] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}
}

__global__ void EngineExecute(const uint n, KernelArray<RayJob*> job, const uint raySeed, const Scene* scene){

	uint index = getGlobalIdx_1D_1D();

	uint startIndex = 0;
	int cur = GetCurrentJob(job, index, startIndex);


	//thrust::default_random_engine rng(randHash(raySeed) * randHash(index));
	//thrust::uniform_real_distribution<float> uniformDistribution(0.0f, 1.0f);


	curandState randState;
	curand_init(raySeed + index, 0, 0, &randState);


	//float prob = uniformDistribution(rng);
	float prob =  curand_uniform(&randState);
	if (index < n){
		
		uint localIndex = index - startIndex / glm::ceil(job[cur]->GetSampleAmount());


			Ray ray;
			job[cur]->GetCamera()->SetupRay(localIndex, ray, randState);


			//calculate something
			glm::vec3 col = scene->IntersectColour(ray) / glm::ceil(job[cur]->GetSampleAmount());



			glm::vec4* pt = &((glm::vec4*)job[cur]->GetResultPointer(0))[localIndex];

			atomicAdd(&(pt->x), col.x);

			atomicAdd(&(pt->y), col.y);

			atomicAdd(&(pt->z), col.z);

	}
}

__host__ void ClearResults(std::vector<RayJob*>& jobs){
	CudaCheck(cudaDeviceSynchronize());
	if (jobs.size() > 0){

		uint n = 0;
		for (int i = 0; i < jobs.size(); i++){
			n += jobs[i]->GetRayAmount()* glm::ceil(jobs[i]->GetSampleAmount());
		}

		if (n != 0){

			thrust::device_vector<RayJob*> deviceJobList(jobs);

			uint blockSize = 64;
			uint gridSize = (n + blockSize - 1) / blockSize;


			//execute engine
			jobL = KernelArray<RayJob*>(deviceJobList);

			cudaEvent_t start, stop;
			float time;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			EngineResultClear << <gridSize, blockSize >> >(n, jobL);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			std::cout << "RayClear Execution: " << time << "ms" << std::endl;
		}
		CudaCheck(cudaDeviceSynchronize());
	}
}
__host__ void ProcessJobs(std::vector<RayJob*>& jobs, const Scene* scene){
	CudaCheck(cudaDeviceSynchronize());
	if (jobs.size()>0){

	uint n = 0;
	for (int i = 0; i < jobs.size();i++ ){
		n += jobs[i]->GetRayAmount()* glm::ceil(jobs[i]->GetSampleAmount());
	}

	if (n!=0){

		thrust::device_vector<RayJob*> deviceJobList(jobs);

		uint blockSize = 64;
		uint gridSize = (n + blockSize - 1) / blockSize;


		//execute engine
		jobL = KernelArray<RayJob*>(deviceJobList);

		cudaEvent_t start, stop; 
		float time;
		cudaEventCreate(&start); 
		cudaEventCreate(&stop); 
		cudaEventRecord(start, 0);


		

		EngineExecute << <gridSize, blockSize >> >(n, jobL, WangHash(raySeedGl++), scene);

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