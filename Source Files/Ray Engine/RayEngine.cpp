#include "RayEngine.h"
#include "CUDA/RayEngine.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility\Timer.h"
#include <deque>

/* List of jobs */
static std::list<RayJob*> jobList;

/* The render derivatives */
static std::deque<double> renderDerivatives;
/* The old render time */
static double oldRenderTime;

/* The frame hold */
static uint frameHold = 5;

/* timer. */
static Timer renderTimer;

/*
 *    Updates the jobs.
 *    @param 		 	renderTime	The render time.
 *    @param 		 	targetTime	The target time.
 *    @param [in,out]	jobs	  	[in,out] If non-null, the jobs.
 */

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

/*
 *    Process this object.
 *    @param	scene 	The scene.
 *    @param	target	Target for the.
 */

void RayEngine::Process(const Scene* scene, double target) {

	//start the timer once actual data movement and calculation starts
	renderTimer.Reset();

	ProcessJobs(jobList, scene);

	double renderTime = renderTimer.Elapsed();

	UpdateJobs(renderTime / 1000.0, target, jobList);
}

/*
 *    Adds a job.
 *    @param 		 	whatToGet	The what to get.
 *    @param 		 	rayAmount	The ray amount.
 *    @param 		 	canChange	True if this object can change.
 *    @param 		 	samples  	The samples.
 *    @param [in,out]	camera   	The camera.
 *    @param [in,out]	resultsIn	If non-null, the results in.
 *    @param [in,out]	extraData	If non-null, information describing the extra.
 *    @return	Null if it fails, else a pointer to a RayJob.
 */

RayJob* RayEngine::AddJob(rayType whatToGet, uint rayAmount, bool canChange,
	float samples, Camera& camera, void* resultsIn, int* extraData) {

	RayJob* job = new RayJob(whatToGet, rayAmount, canChange, samples, camera, resultsIn, extraData);
	jobList.push_back(job);

	return job;
}

/*
 *    Modify job.
 *    @param [in,out]	jobIn 	If non-null, the job in.
 *    @param [in,out]	camera	The camera.
 */

void RayEngine::ModifyJob(RayJob* jobIn, Camera& camera) {

	for (auto& job : jobList) {
		if (job == jobIn) {
			job->camera = camera;
			break;
		}
	}

}

/*
 *    Removes the job described by job.
 *    @param [in,out]	job	If non-null, the job.
 *    @return	True if it succeeds, false if it fails.
 */

bool RayEngine::RemoveJob(RayJob* job) {

	jobList.remove(job);

	return true;
}
/* Initializes this object. */
void RayEngine::Initialize() {
	GPUInitialize();
}
/* Terminates this object. */
void RayEngine::Terminate() {
	GPUTerminate();
}