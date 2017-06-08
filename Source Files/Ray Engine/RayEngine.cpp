#include "RayEngine.h"
#include "CUDA/RayEngine.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility\Timer.h"


static std::list<RayJob*> jobList;

//timer
static Timer timer;

void UpdateJobs(double in, double target, std::list<RayJob*>& jobs) {

	double ratio = target / in;

	for (auto& job : jobs) {
		if (job->canChange) {
			job->samples *= ratio;
		}
	}

}

void RayEngine::Process(const Scene* scene){

	//start the timer once acctual dat movement and calculation starts
	timer.Reset();

	ProcessJobs(jobList, scene);

	UpdateJobs(timer.Elapsed(), 14.0, jobList);

}

RayJob* RayEngine::AddJob(rayType whatToGet, uint rayAmount, bool canChange,
	float samples, Camera& camera, void* resultsIn, int* extraData) {

	RayJob* job= new RayJob(whatToGet, rayAmount, canChange, samples, camera, resultsIn, extraData);
	jobList.push_back(job);

	return job;
}

void RayEngine::ModifyJob(RayJob* jobIn, Camera& camera) {

	for (auto& job : jobList) {
		if (job== jobIn) {
			job->camera = camera;
			break;
		}
	}

}

bool RayEngine::RemoveJob(RayJob* job){

	jobList.remove(job);

	return true;
}
void RayEngine::Initialize() {
	GPUInitialize();
}
void RayEngine::Terminate(){
	GPUTerminate();
}