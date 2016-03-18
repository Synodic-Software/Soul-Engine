#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Engine Core\Scene\Scene.h"
#include "RayJob.h"
#include <thrust/device_vector.h>


//The main engine that processes RayJobs
namespace RayEngine {

	//Adds a job to be executed after all updates have taken place. It will execute the 
	//given function to initialize all its rays and returns the tag that can be used to extract the data in 'UpdateLate'
	RayJob* AddRayJob(rayType, uint, uint, Camera*,uint);

	//A varient that does not copy the results to the CPU but instead returns a cuda* that can be procesed further.
	//adds a job with a hint to keep its allocated data for ray storage. Speed gains if large ray bundles are given.
	//RayJob* AddRecurringRayJob(rayType, uint, uint, Camera*);

	//amount cant be changed as it effects storage, but all else can;
	//return false if it wasn't changed
	bool ChangeJob(RayJob*, uint, float, Camera*);

	bool SwapResults(RayJob*, uint, uint);

	void Process(const Scene*);
	void Clear();
	void Clean();

}