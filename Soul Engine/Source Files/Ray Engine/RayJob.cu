#include "RayJob.cuh"

__host__ RayJob::RayJob(castType whatToGet, uint rayAmountN, uint newSamples, Camera* cameraN, bool isRecurringN){

	probability = 1.0f;

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

__host__ RayJob::~RayJob(){
	if (resultsF!=NULL){
		delete resultsF;
	}
	if (resultsI != NULL){
		delete resultsI;
	}
	if (resultsT != NULL){
		delete resultsT;
	}
}

CUDA_FUNCTION void RayJob::ChangeProbability(float newProb){

	probability = newProb;

}

CUDA_FUNCTION float RayJob::GetProbability(){

	return probability;

}