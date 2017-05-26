#include "RayEngine.h"
#include "CUDA/RayEngine.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"


static std::list<RayJob*> jobList;

void RayEngine::Process(const Scene* scene){
	ProcessJobs(jobList, scene);
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