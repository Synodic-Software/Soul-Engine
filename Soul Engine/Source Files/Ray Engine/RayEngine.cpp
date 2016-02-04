#include "RayEngine.h"
#include "CUDA/RayEngine.cuh"


std::vector<RayJob*> jobList;


void RayEngine::Process(const Scene* scene){
	ProcessJobs(jobList, scene);
}

RayJob* RayEngine::AddRayJob(rayType whatToGet, uint rayAmount,
	uint samples, Camera* camera){

	RayJob* newJob = new RayJob(whatToGet, rayAmount, samples, camera);
	jobList.push_back(newJob);

	return newJob;
}

bool RayEngine::ChangeJob(RayJob* job, uint rayAmount, 
	uint samples, Camera* camera){

	if (rayAmount<=job->RayAmountMax()){
		job->GetRayAmount() = rayAmount;
		job->GetSampleAmount() = samples;
		job->GetCamera() = camera;

		return true;
	}
	else{
		return false;
	}

}