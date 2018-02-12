#include "RayJob.cuh"

static uint counter = 0;

__host__ RayJob::RayJob(rayType whatToGet, bool _canChange, float newSamples)
{

	type = whatToGet;
	samples = newSamples;
	rayOffset = 0;
	canChange = _canChange;
	id = counter++;
}

__host__ RayJob::RayJob()
{

}

__host__ RayJob::~RayJob() {

}