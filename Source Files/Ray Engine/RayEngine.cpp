#include "RayEngine.h"
#include "CUDA/RayEngine.cuh"

#include "Utility/Timer.h"
#include <deque>
#include "Algorithms/Filters/Filter.h"
#include "Compute/GPUManager.h"

/* List of jobs */
static ComputeBuffer<RayJob> jobList;

/* The render derivatives */
static std::deque<double> renderDerivatives;

/* The old render time */
static double oldRenderTime;

/* The frame hold */
static uint frameHold = 5;

/* timer. */
static Timer renderTimer;

static ComputeBuffer<Ray> deviceRaysA;
static ComputeBuffer<Ray> deviceRaysB;

static ComputeBuffer<curandState> randomState;

static uint raySeedGl = 0;

static const uint rayDepth = 4;

//stored counters
static ComputeBuffer<int> counter;
static ComputeBuffer<int> hitAtomic;

static GPUExecutePolicy persistantPolicy;

__host__ __device__ __inline__ uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

/*
 *    Updates the jobs.
 *    @param 		 	renderTime	The render time.
 *    @param 		 	targetTime	The target time.
 *    @param [in,out]	jobs	  	[in,out] If non-null, the jobs.
 */

void UpdateJobs(double renderTime, double targetTime, ComputeBuffer<RayJob>& jobs) {

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
	for (int i = 0; i < jobList.SizeHost(); i++) {

		RayJob& job = jobList[i];
		if (false) {

			Camera& camera = job.camera;

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

			//float tempSamples = job->samples * change;

			//ComputeBuffer<Camera>* cameraBuffer = CameraManager::GetCameraBuffer();

			//Camera& camera = (*cameraBuffer)[job->camera];

			//if (tempSamples < 1.0f) {

			//	job->samples = 1.0f;
			//	float value = (1.0f - (job->samples - 1.0f) / (job->samples - tempSamples)) * change;
			//	camera.film.resolution.x = camera.film.resolutionMax.x * value;
			//	camera.film.resolution.y = camera.film.resolutionMax.y * value;
			//}
			//else {
			//	if (camera.film.resolution != camera.film.resolutionMax) {
			//		//job->camera.film.resolution *= change*countChange + 1.0;
			//	}
			//	job->samples = tempSamples;
			//}

		}
	}
}

/*
 *    Process this object.
 *    @param	scene 	The scene.
 *    @param	target	Target for the.
 */

void RayEngine::Process(std::vector<Scene>& scene, double target) {

	//start the timer once actual data movement and calculation starts
	renderTimer.Reset();

	auto numberJobs = jobList.SizeHost();

	//only upload data if a job exists
	if (numberJobs > 0) {

		uint numberResults = 0;
		uint numberRays = 0;

		for (uint i = 0; i < numberJobs; ++i) {

			const Camera camera = jobList[i].camera;

			const uint rayAmount = camera.film.resolution.x*camera.film.resolution.y;

			jobList[i].rayOffset = numberResults;
			numberResults += rayAmount;

			if (jobList[i].samples < 0) {
				jobList[i].samples = 0.0f;
			}

			numberRays += rayAmount* uint(glm::ceil(jobList[i].samples));

		}

		if (numberResults != 0 && numberRays != 0) {

			jobList.TransferToDevice();

			//clear the jobs result memory, required for accumulation of multiple samples

			GPUDevice device = GPUManager::GetBestGPU();

			const uint blockSize = 64;
			const GPUExecutePolicy normalPolicy(glm::vec3((numberResults + blockSize - 1) / blockSize,1,1), glm::vec3(blockSize,1,1), 0, 0);

			auto jobP = jobList.DataDevice();
			device.Launch(normalPolicy, EngineSetup, numberResults, jobP, numberJobs);

			if (numberRays > deviceRaysA.SizeDevice()) {

				randomState.Resize(numberRays);
				deviceRaysA.Resize(numberRays);
				deviceRaysB.Resize(numberRays);

				auto rand = WangHash(++raySeedGl);
				auto randP = randomState.DataDevice();
				device.Launch(normalPolicy, RandomSetup, numberRays, randP, rand);

			}

			//TODO handle multiple scenes
			//copy the scene data over
			scene[0].faces.TransferToDevice();
			scene[0].vertices.TransferToDevice();
			scene[0].materials.TransferToDevice();
			scene[0].tets.TransferToDevice();
			scene[0].objects.TransferToDevice();

			scene[0].sky.TransferToDevice();

			//setup the counters
			counter[0] = 0;
			hitAtomic[0] = 0;

			counter.TransferToDevice();
			hitAtomic.TransferToDevice();

			auto raysAP = deviceRaysA.DataDevice();
			auto hitP = hitAtomic.DataDevice();
			auto randP = randomState.DataDevice();

			device.Launch(normalPolicy, RaySetup, numberRays, numberJobs, jobP, raysAP, hitP, randP);

			//start the engine loop
			hitAtomic.TransferToHost();

			uint numActive = hitAtomic[0];

			for (uint i = 0; i < rayDepth && numActive>0; ++i) {

				//reset counters
				counter[0] = 0;
				hitAtomic[0] = 0;

				counter.TransferToDevice();
				hitAtomic.TransferToDevice();

				//grab the current block sizes for collecting hits based on numActive
				const GPUExecutePolicy activePolicy(glm::vec3((numActive + blockSize - 1) / blockSize, 1, 1), glm::vec3(blockSize, 1, 1), 0, 0);

				raysAP = deviceRaysA.DataDevice();
				auto dataP = scene[0].bvhData.DataDevice();
				auto vertP = scene[0].vertices.DataDevice();
				auto faceP = scene[0].faces.DataDevice();
				auto counterP = counter.DataDevice();

				//main engine, collects hits
				device.Launch(persistantPolicy, ExecuteJobs, numActive, raysAP, dataP, vertP, faceP, counterP);

				auto raysBP = deviceRaysB.DataDevice();
				auto skyP = scene[0].sky.DataDevice();
				auto matP = scene[0].materials.DataDevice();

				//processes hits 
				device.Launch(activePolicy, ProcessHits, numActive, jobP, numberJobs, raysAP, raysBP, skyP, faceP, vertP, matP, hitP, randP);

				std::swap(deviceRaysA, deviceRaysB);


				hitAtomic.TransferToHost();

				numActive = hitAtomic[0];
			}
		}

	}

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

	jobList.PushBack(RayJob(whatToGet, canChange, samples));

	return jobList[jobList.SizeHost() - 1].id;
}

/*
 *    Modify job.
 *    @param [in,out]	jobIn 	If non-null, the job in.
 *    @param [in,out]	camera	The camera.
 */

RayJob& RayEngine::GetJob(uint jobIn) {

	for (int i = 0; i < jobList.SizeHost(); i++) {

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

	deviceRaysA.Move(GPUManager::GetBestGPU());
	deviceRaysB.Move(GPUManager::GetBestGPU());

	randomState.Move(GPUManager::GetBestGPU());

	counter.Move(GPUManager::GetBestGPU());
	hitAtomic.Move(GPUManager::GetBestGPU());

	jobList.Move(GPUManager::GetBestGPU());

	counter.Resize(1);
	hitAtomic.Resize(1);
	
	persistantPolicy = GPUManager::GetBestGPU().BestExecutePolicy(ExecuteJobs);


	///////////////Alternative Hardcoded Calculation/////////////////
	//uint blockPerSM = CUDABackend::GetBlocksPerMP();
	//warpPerBlock = CUDABackend::GetWarpsPerMP() / blockPerSM;
	//blockCountE = CUDABackend::GetSMCount()*blockPerSM;
	//blockSizeE = dim3(CUDABackend::GetWarpSize(), warpPerBlock, 1);

}

void RayEngine::Update() {
	for (int i = 0; i < jobList.SizeHost(); i++) {

		RayJob& job = jobList[i];
		job.camera.UpdateVariables();
	}
}

/* Terminates this object. */
void RayEngine::Terminate() {

}

void RayEngine::PreProcess() {

}
void RayEngine::PostProcess() {

	//grab job pointer
	auto job = *jobList.begin();

	//grab camera
	Camera camera = job.camera;

	//Filter::IterativeBicubic((glm::vec4*)job.camera.film.results, camera.film.resolution, camera.film.resolutionMax);
}
