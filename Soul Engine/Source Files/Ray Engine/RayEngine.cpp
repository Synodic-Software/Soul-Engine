#include "RayEngine.h"

RayEngine::RayEngine(){

}
void RayEngine::Process(){
	ProcessJobs();
}

void RayEngine::AddRayJob(castType whatToGet, uint typeAmount, RayFunction setupFunction, uint rayAmount){
	jobs.emplace_back(new RayJob(whatToGet, typeAmount, setupFunction, rayAmount));
}