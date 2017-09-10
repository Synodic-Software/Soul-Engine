#include "RayJob.cuh"

static uint counter=0;

__host__ RayJob::RayJob(rayType whatToGet, uint rayAmountN, bool _canChange, float newSamples, uint _camera, void* resultsIN, int* extraData) {

	type = whatToGet;
	rayAmount = rayAmountN;
	rayBaseAmount = rayAmount;
	samples = newSamples;
	camera = _camera;
	results = resultsIN;
	startIndex = 0;
	groupData = extraData;
	canChange = _canChange;
	ID = counter++;
}
__host__ RayJob::RayJob() {

}
__host__ RayJob::~RayJob() {

}