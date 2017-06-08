#include "RayEngine.h"
#include "CUDA/RayEngine.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility\Timer.h"
#include <deque>

static std::list<RayJob*> jobList;

static std::deque<double> derivatives;
static double oldTime;

static uint frameHold = 5;

//timer
static Timer timer;

void UpdateJobs(double in, double target, std::list<RayJob*>& jobs) {

	//if it the first frame, pass the target as the '0th' frame
	if (derivatives.size() == 0) {
		oldTime = target;
	}

	//count the jobs that can be modified
	int count = 0;
	for (auto& job : jobs) {
		if (job->canChange) {
			count++;
		}
	}

	//disperse equal cuts to all of them
	double countChange = 1.0f / count;

	//push the new derivative
	derivatives.push_front(oldTime - in);
	oldTime = in;

	//cull the frame counts
	if (derivatives.size() > frameHold) {
		derivatives.pop_back();
	}

	//calculate the average derivative
	double averageDV = 0.0;
	for (auto& dv : derivatives) {
		averageDV += dv / derivatives.size();
	}

	//use the average derivative to grab an expected next frametime
	double expected = in + averageDV;

	double change = (target / expected);

	//modify all the sample counts to reflect the change
	for (auto& job : jobs) {
		if (job->canChange) {
			job->samples *= change*countChange;
		}
	}
}

void RayEngine::Process(const Scene* scene) {

	//start the timer once acctual dat movement and calculation starts
	timer.Reset();

	ProcessJobs(jobList, scene);

	UpdateJobs(timer.Elapsed(), 16.6666, jobList);

}

RayJob* RayEngine::AddJob(rayType whatToGet, uint rayAmount, bool canChange,
	float samples, Camera& camera, void* resultsIn, int* extraData) {

	RayJob* job = new RayJob(whatToGet, rayAmount, canChange, samples, camera, resultsIn, extraData);
	jobList.push_back(job);

	return job;
}

void RayEngine::ModifyJob(RayJob* jobIn, Camera& camera) {

	for (auto& job : jobList) {
		if (job == jobIn) {
			job->camera = camera;
			break;
		}
	}

}

bool RayEngine::RemoveJob(RayJob* job) {

	jobList.remove(job);

	return true;
}
void RayEngine::Initialize() {
	GPUInitialize();
}
void RayEngine::Terminate() {
	GPUTerminate();
}