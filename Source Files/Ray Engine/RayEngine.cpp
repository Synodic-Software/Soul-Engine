#include "RayEngine.h"
#include "CUDA/RayEngine.cuh"
#include "Utility/CUDA/CUDAHelper.cuh"
#include "Utility/Timer.h"
#include <deque>
#include "Algorithms/Filters/Filter.h"
#include "GPGPU/GPUManager.h"

/* List of jobs */
static GPUBuffer<RayJob> jobList;

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

void UpdateJobs(double renderTime, double targetTime, GPUBuffer<RayJob>& jobs) {

	//if it the first frame, pass the target as the '0th' frame
	if (renderDerivatives.size() == 0) {
		oldRenderTime = renderTime;
	}

	//count the jobs that can be modified
	int count = 0;
	for (auto& job : jobs) {
		if (job.canChange) {
			count++;
		}
	}

	//disperse equal cuts to all of them
	double countChange = 1.0f / count;

	//push the new derivative
	renderDerivatives.push_front(oldRenderTime - renderTime);
	oldRenderTime = renderTime;

	//cull the frame counts
	if (renderDerivatives.size() > frameHold) {
		renderDerivatives.pop_back();
	}

	//calculate the average derivative
	double averageRenderDV = 0.0;

	for (auto itrR = renderDerivatives.begin(); itrR != renderDerivatives.end(); ++itrR) {
		averageRenderDV += *itrR;
	}

	averageRenderDV /= renderDerivatives.size();

	//use the average derivative to grab an expected next frametime
	double expectedRender = renderTime + averageRenderDV;

	//target time -5% to account for frame instabilities and try to stay above target
	//double change = (targetTime / expectedRender - 1.0) * 0.95f * countChange + 1.0;

	double change = (targetTime - renderTime) / targetTime;


	//modify all the sample counts/ resolutions to reflect the change
	for (int i = 0; i < jobList.size(); i++) {

		RayJob& job = jobList[i];

		Camera& camera = job.camera;

		//set the previous resolution
		camera.film.resolutionPrev = camera.film.resolution;

		float delta = change*camera.film.resolutionRatio;
		float newRatio = camera.film.resolutionRatio + delta;

		if (camera.film.resolutionMax.x * newRatio < 64) {

			newRatio = 64 / (float)camera.film.resolutionMax.x;

		}

		if (newRatio >= 1.0f) {

			camera.film.resolution = camera.film.resolutionMax;
			job.samples = newRatio;

		}
		else {

			camera.film.resolution.x = camera.film.resolutionMax.x * newRatio;
			camera.film.resolution.y = camera.film.resolutionMax.y * newRatio;
			job.samples = 1.0f;

		}

		//update the camera ratio
		camera.film.resolutionRatio = newRatio;

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

	UpdateJobs(renderTimer.Elapsed() / 1000.0, target, jobList);
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

uint RayEngine::AddJob(rayType whatToGet, bool canChange,
	float samples) {

	jobList.push_back(RayJob(whatToGet, canChange, samples));

	return jobList[jobList.size() - 1].id;
}

/*
 *    Modify job.
 *    @param [in,out]	jobIn 	If non-null, the job in.
 *    @param [in,out]	camera	The camera.
 */

RayJob& RayEngine::GetJob(uint jobIn) {

	for (int i = 0; i < jobList.size(); i++) {

		RayJob& job = jobList[i];
		if (job.id == jobIn) {
			return job;
		}
	}

}

/*
 *    Removes the job described by job.
 *    @param [in,out]	job	If non-null, the job.
 *    @return	True if it succeeds, false if it fails.
 */

bool RayEngine::RemoveJob(uint job) {
	//TODO implement
	//jobList.remove(job);

	return true;
}

/* Initializes this object. */
void RayEngine::Initialize() {

	GPUInitialize();

	jobList.TransferDevice(GPUManager::GetBestGPU());

}

void RayEngine::Update() {
	for (int i = 0; i < jobList.size(); i++) {

		RayJob& job = jobList[i];
		job.camera.UpdateVariables();
	}
}

/* Terminates this object. */
void RayEngine::Terminate() {
	GPUTerminate();
}

void RayEngine::PreProcess() {

}

void RayEngine::PostProcess() {

	//grab job pointer
	auto job = *jobList.begin();

	//grab camera
	Camera camera = job.camera;

	//Filter::IterativeBicubic((glm::vec4*)job.camera.film.results, camera.film.resolution, camera.film.resolutionMax);
	Filter::Nearest((glm::vec4*)job.camera.film.results, camera.film.resolutionMax, camera.film.resolutionPrev);
}
