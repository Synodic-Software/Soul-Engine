#include "RayJob.cuh"

__host__ RayJob::RayJob(rayType whatToGet, uint rayAmountN, uint newSamples, Camera* cameraN, uint numResultBuffersN){

	type = whatToGet;
	rayAmount = rayAmountN;
	rayBaseAmount = rayAmount;
	samples = newSamples;
	camera = cameraN;
	numResultBuffers = numResultBuffersN;

	CudaCheck(cudaMallocManaged((void**)&results, numResultBuffers*sizeof(glm::vec4*)));
	for (int i = 0; i < numResultBuffers; i++){
		CudaCheck(cudaMallocManaged((void**)&results[i], rayBaseAmount*sizeof(glm::vec4)));
	}

}

__host__ RayJob::~RayJob(){
	if (results != NULL){
		for (int i = 0; i < numResultBuffers; i++){
			if (results[i] != NULL){
				cudaFree(results[i]);
			}
		}
		cudaFree(results);
	}
}

//Returns a reference to a camera pointer. All the ray shooting information is stored here.
CUDA_FUNCTION Camera*& RayJob::GetCamera(){
	return camera;
}

//Returns the rayType of the job.
CUDA_FUNCTION rayType RayJob::RayType() const{
	return type;
}

//Returns the Ray max of the job as per its initialization params.
CUDA_FUNCTION uint RayJob::RayAmountMax() const{
	return rayBaseAmount;
}

//Returns the current rayAmount (modifiable)
CUDA_FUNCTION uint& RayJob::GetRayAmount() {
	return rayAmount;
}

//Returns the current sample per ray (modifiable)
CUDA_FUNCTION uint& RayJob::GetSampleAmount() {
	return samples;
}

//Returns the pointer to the results (modifiable)
CUDA_FUNCTION void*& RayJob::GetResultPointer(uint x){
	return results[x];
}

CUDA_FUNCTION void RayJob::SwapResults(uint a, uint b){
	void* temp = results[a];
	results[a] = results[b];
	results[b] = temp;
}
