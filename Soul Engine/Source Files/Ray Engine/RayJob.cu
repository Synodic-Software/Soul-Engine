#include "RayJob.cuh"

RayJob::RayJob(castType whatToGet, uint rayAmountN, uint newSamples, Camera* cameraN, bool isRecurringN){

	type = whatToGet;
	rayAmount = rayAmountN;
	rayBaseAmount = rayAmount;
	samples = newSamples;
	camera = cameraN;
	isRecurring = isRecurringN;
	nextRay = NULL;

	if (whatToGet != RayOBJECT_ID&&whatToGet!=RayCOLOUR_TO_BUFFER){
		cudaMallocManaged(&resultsF, rayBaseAmount);
		resultsI = NULL;
		resultsT = NULL;
	}
	else if (whatToGet==RayCOLOUR_TO_BUFFER){

		

		resultsI = NULL;
		resultsF = NULL;
	}
	else{
		cudaMallocManaged(&resultsI, rayBaseAmount);
		resultsF = NULL;
		resultsT = NULL;
	}
}