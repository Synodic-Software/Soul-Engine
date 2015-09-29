#include "RayJob.cuh"

RayJob::RayJob(castType whatToGet, RayFunction setupFunction, uint rayAmount, uint newSamples, bool isR, float3 forwardN, float3 rightN, float3 oriN, float distN, float2 fovN){

	isReaccuring = isR;
	type = whatToGet;
	rayAmount = rayAmount;
	rayBaseAmount = rayAmount;
	raySetup = setupFunction;
	samples = newSamples;
	forward = forwardN;
	right = rightN;
	origin = oriN;
	distanceFromO = distN;
	fov = fovN;


	if (whatToGet!=RayOBJECT_ID){
		cudaMallocManaged(&resultsF, rayBaseAmount);
		resultsI = NULL;
	}
	else{
		cudaMallocManaged(&resultsI, rayBaseAmount);
		resultsF = NULL;
	}
}