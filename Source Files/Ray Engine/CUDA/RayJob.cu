#include "RayJob.cuh"
#include "Utility/CUDA/CUDAHelper.cuh"

__host__ RayJob::RayJob(rayType whatToGet, uint rayAmountN, float newSamples, Camera cameraN, void* resultsIN,int* extraData) {

	type = whatToGet;
	rayAmount = rayAmountN;
	rayBaseAmount = rayAmount;
	samples = newSamples;
	camera = cameraN;
	results = resultsIN;
	startIndex = 0;
	groupData = extraData;

}

__host__ RayJob::~RayJob() {

}