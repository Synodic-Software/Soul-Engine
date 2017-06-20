#include "RayEngine.h"
#include "CUDA/RayEngine.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility\Timer.h"
#include <deque>

static std::list<RayJob*> jobList;

static std::deque<double> renderDerivatives;
static double oldRenderTime;

static uint frameHold = 5;

//timer
static Timer renderTimer;

void UpdateJobs(double renderTime, double targetTime, std::list<RayJob*>& jobs) {

	//if it the first frame, pass the target as the '0th' frame
	if (renderDerivatives.size() == 0) {
		oldRenderTime = renderTime;
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
	renderDerivatives.push_front(oldRenderTime- renderTime);
	oldRenderTime = renderTime;

	//cull the frame counts
	if (renderDerivatives.size() > frameHold) {
		renderDerivatives.pop_back();
	}

	//calculate the average derivative
	double averageRenderDV = 0.0;

	for (auto itrR = renderDerivatives.begin(); itrR != renderDerivatives.end(); itrR++) {
		averageRenderDV += *itrR;
	}

	averageRenderDV /= renderDerivatives.size();

	//use the average derivative to grab an expected next frametime
	double expectedRender = renderTime + averageRenderDV;

	//target time -5% to account for frame instabilities and consitantly stay above target
	double change = targetTime / expectedRender - 1.0;

	//modify all the sample counts to reflect the change
	for (auto& job : jobs) {
		if (job->canChange) {
			job->samples *= change*countChange + 1.0;
		}
	}
}

void RayEngine::Process(const Scene* scene, double target) {

	//start the timer once actual data movement and calculation starts
	renderTimer.Reset();

	ProcessJobs(jobList, scene);

	double renderTime = renderTimer.Elapsed();

	UpdateJobs(renderTime / 1000.0, target, jobList);
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