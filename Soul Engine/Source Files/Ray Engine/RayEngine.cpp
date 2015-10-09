#include "RayEngine.h"
#include "RayEngine.cuh"

RayJob* jobs=NULL;
uint jobSize=0;

void RayEngine::Process(){
	ProcessJobs(jobs);
}

RayJob* RayEngine::AddRayJob(castType whatToGet, uint rayAmount,
	uint samples, Camera* camera){

	RayJob* temp = jobs;
	RayJob* newHead = new RayJob(whatToGet, rayAmount, samples, camera, false);
	newHead->nextRay = temp;
	jobs = newHead;
	jobSize++;

	return newHead;
}

RayJob* RayEngine::AddRecurringRayJob(castType whatToGet, 
	uint rayAmount, uint samples, Camera* camera){

	RayJob* temp = jobs;
	RayJob* newHead = new RayJob(whatToGet, rayAmount, samples, camera, true);
	newHead->nextRay = temp;
	jobs = newHead;
	jobSize++;

	return newHead;
}

bool RayEngine::ChangeJob(RayJob* job, uint rayAmount, 
	uint samples, Camera* camera){

	if (job->IsRecurring() && rayAmount<=job->rayBaseAmount){
		job->rayAmount = rayAmount;
		job->samples = samples;
		job->camera = camera;

		return true;
	}
	else{
		return false;
	}

}