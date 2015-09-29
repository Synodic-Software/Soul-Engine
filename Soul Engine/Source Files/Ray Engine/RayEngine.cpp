#include "RayEngine.h"
#include "RayEngine.cuh"

RayJob* jobs=NULL;
uint jobSize=0;

void RayEngine::Process(){
	ProcessJobs(jobs);
}

RayJob* RayEngine::AddRayJob(castType whatToGet, RayFunction setupFunction, uint rayAmount, uint samples, float3 forward, float3 right, float3 ori, float dist, float2 fov){
	RayJob* temp = jobs;
	RayJob* newHead = new RayJob(whatToGet, setupFunction, rayAmount, samples, false, forward, right, ori, dist, fov);
	newHead->nextRay = temp;
	jobSize++;
}

RayJob* RayEngine::AddRecurringRayJob(castType whatToGet, RayFunction setupFunction, uint rayAmount, uint samples, float3 forward, float3 right, float3 ori, float dist, float2 fov){
	RayJob* temp = jobs;
	RayJob* newHead = new RayJob(whatToGet, setupFunction, rayAmount, samples, true, forward, right, ori, dist, fov);
	newHead->nextRay = temp;
	jobSize++;
}

bool RayEngine::ChangeJob(RayJob* job, RayFunction setupFunction, uint rayAmount, uint samples, float3 forward, float3 right, float3 ori, float dist, float2 fov){
	if (job->IsReaccuring() && rayAmount<=job->rayBaseAmount){
		job->raySetup = setupFunction;
		job->rayAmount = rayAmount;
		job->samples = samples;
		job->forward = forward;
		job->right = right;
		job->origin = ori;
		job->distanceFromO = dist;
		job->fov = fov;
	}
	else{
		return false;
	}
}