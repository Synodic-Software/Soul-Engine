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