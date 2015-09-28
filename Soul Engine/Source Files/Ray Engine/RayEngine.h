#pragma once

#include "Engine Core/BasicDependencies.h"
#include "RayJob.cuh"


//The main engine that processes RayJobs
class RayEngine {
public:

	RayEngine();

	//Adds a job to be executed after all updates have taken place. It will execute the 
	//given function to initialize all its rays and returns the tag that can be used to extract the data in 'UpdateLate'
	void AddRayJob(castType, uint, RayFunction, uint);

	//A varient that does not copy the results to the CPU but instead returns a cuda* that can be procesed further.
	//adds a job with a hint to keep its allocated data for ray storage. Speed gains if large ray bundles are given.
	void AddRecurringRayJob(castType, uint, RayFunction, uint);

	void Process();
	__global__ void ProcessJobs();
private:
	//vector of all the jobs to execute this frame.
	std::vector<RayJob*> jobs;

};
