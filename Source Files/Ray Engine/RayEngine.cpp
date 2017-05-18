#include "RayEngine.h"
#include "CUDA/RayEngine.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"


std::vector<RayJob> jobList;

void RayEngine::Process(const Scene* scene){
	ProcessJobs(jobList, scene);
}

void RayEngine::AddRayJob(rayType whatToGet, uint rayAmount,
	float samples, Camera& camera, void* resultsIn, int* extraData) {

	jobList.push_back({ whatToGet, rayAmount, samples, camera, resultsIn,extraData });

}

bool RayEngine::ChangeJob(RayJob* job, uint rayAmount, 
	float samples, Camera& camera){

	if (rayAmount<=job->rayBaseAmount){
		job->rayAmount = rayAmount;
		job->samples = samples;
		job->camera = camera;

		return true;
	}
	else{
		return false;
	}

}
void RayEngine::Initialize() {
	GPUInitialize();
}
void RayEngine::Terminate(){
	GPUTerminate();
}