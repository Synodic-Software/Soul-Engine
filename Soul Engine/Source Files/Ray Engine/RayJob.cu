#include "RayJob.cuh"

RayJob::RayJob(castType whatToGet, uint rayAmountN, uint newSamples, Camera* cameraN, bool isRecurringN){

	type = whatToGet;
	rayAmount = rayAmountN;
	rayBaseAmount = rayAmount;
	samples = newSamples;
	camera = cameraN;
	isRecurring = isRecurringN;
	nextRay = NULL;

	if (whatToGet != RayOBJECT_ID&&!RayCOLOUR_TO_TEXTURE){
		cudaMallocManaged(&resultsF, rayBaseAmount);
		resultsI = NULL;
		resultsT = NULL;
	}
	else if (RayCOLOUR_TO_TEXTURE){

		

		resultsI = NULL;
		resultsF = NULL;
	}
	else{
		cudaMallocManaged(&resultsI, rayBaseAmount);
		resultsF = NULL;
		resultsT = NULL;
	}
}