#pragma once

#include "Engine Core/BasicDependencies.h"
#include "RayJob.cuh"



//The main engine that processes RayJobs
namespace RayEngine {

	//Adds a job to be executed after all updates have taken place. It will execute the 
	//given function to initialize all its rays and returns the tag that can be used to extract the data in 'UpdateLate'
	RayJob* AddRayJob(castType, RayFunction, uint, uint, float3, float3, float3, float, float2);

	//A varient that does not copy the results to the CPU but instead returns a cuda* that can be procesed further.
	//adds a job with a hint to keep its allocated data for ray storage. Speed gains if large ray bundles are given.
	RayJob* AddRecurringRayJob(castType, RayFunction, uint, uint, float3, float3, float3, float, float2);

	//amount cant be changed as it effects storage, but all else can;
	//return false if it wasn't changed
	bool ChangeJob(RayJob*, RayFunction, uint, uint, float3, float3, float3, float, float2);

	void Process();


}