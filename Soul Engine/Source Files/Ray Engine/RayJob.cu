#include "RayJob.cuh"

__host__ RayJob::RayJob(rayType whatToGet, uint rayAmountN, float newSamples, Camera* cameraN, bool isRecurringN){

	type = whatToGet;
	rayAmount = rayAmountN;
	rayBaseAmount = rayAmount;
	samples = newSamples;
	camera = cameraN;
	isRecurring = isRecurringN;

	if (whatToGet != RayOBJECT_ID&&whatToGet!=RayCOLOUR){
		cudaMallocManaged(&resultsF, rayBaseAmount);
		resultsI = NULL;
		resultsT = NULL;
	}
	else if (whatToGet==RayCOLOUR){

		

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

//Returns a reference to a camera pointer. All the ray shooting information is stored here.
CUDA_FUNCTION Camera*& RayJob::GetCamera(){
	return camera;
}

//Returns a boolean of the jobs storage flag.
CUDA_FUNCTION bool RayJob::IsRecurring() const{
	return isRecurring;
}

//Returns the rayType of the job.
CUDA_FUNCTION rayType RayJob::RayType() const{
	return type;
}

//Returns the Ray max of the job as per its initialization params.
CUDA_FUNCTION uint RayJob::RayAmountMax() const{
	return rayBaseAmount;
}

//Returns the current rayAmount (modifiable)
CUDA_FUNCTION uint& RayJob::GetRayAmount() {
	return rayAmount;
}

//Returns the current sample per ray (modifiable)
CUDA_FUNCTION float& RayJob::GetSampleAmount() {
	return samples;
}