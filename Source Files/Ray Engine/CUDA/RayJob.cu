#include "RayJob.cuh"
#include "Utility/CUDA/CUDAHelper.cuh"

__host__ RayJob::RayJob(rayType whatToGet, uint rayAmountN, uint newSamples, Camera cameraN, void* resultsIN) {

	type = whatToGet;
	rayAmount = rayAmountN;
	rayBaseAmount = rayAmount;
	samples = newSamples;
	camera = cameraN;
	results = resultsIN;
	startIndex = 0;
}

__host__ RayJob::~RayJob() {

}

//Returns a reference to a camera pointer. All the ray shooting information is stored here.
__host__ __device__ Camera& RayJob::GetCamera() {
	return camera;
}

//Returns the rayType of the job.
__host__ __device__ rayType RayJob::RayType() const {
	return type;
}

//Returns the Ray max of the job as per its initialization params.
__host__ __device__ uint RayJob::RayAmountMax() const {
	return rayBaseAmount;
}

//Returns the current rayAmount (modifiable)
__host__ __device__ uint& RayJob::GetRayAmount() {
	return rayAmount;
}

//Returns the current sample per ray (modifiable)
__host__ __device__ uint& RayJob::GetSampleAmount() {
	return samples;
}
