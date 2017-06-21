#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Engine Core\Scene\Scene.h"
#include "RayJob.h"
#include <thrust/device_vector.h>

//defined in winspool.h
#undef AddJob

//The main engine that processes RayJobs
namespace RayEngine {

	/*
	 *    Adds a job to be executed after all updates have taken place. It will execute the given
	 *    function to initialize all its rays and returns the tag that can be used to extract the
	 *    data in 'UpdateLate'.
	 *
	 *    @param 		 	parameter1	The first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 *    @param 		 	parameter3	True to parameter 3.
	 *    @param 		 	parameter4	The fourth parameter.
	 *    @param [in,out]	parameter5	The fifth parameter.
	 *    @param [in,out]	parameter6	If non-null, the parameter 6.
	 *    @param [in,out]	parameter7	If non-null, the parameter 7.
	 *
	 *    @return	Null if it fails, else a pointer to a RayJob.
	 */

	RayJob* AddJob(rayType, uint, bool,float, Camera&, void*,int*);

	//A varient that does not copy the results to the CPU but instead returns a cuda* that can be procesed further.
	//adds a job with a hint to keep its allocated data for ray storage. Speed gains if large ray bundles are given.
	//RayJob* AddRecurringRayJob(rayType, uint, uint, Camera*);

	/*
	 *    amount cant be changed as it effects storage, but all else can;
	 *    return false if it wasn't changed.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *
	 *    @return	True if it succeeds, false if it fails.
	 */

	bool RemoveJob(RayJob*);

	/*
	 *    Modify job.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param [in,out]	parameter2	The second parameter.
	 */

	void ModifyJob(RayJob*, Camera&);

	/*
	 *    Process this object.
	 *
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 */

	void Process(const Scene*, double);
	/* Initializes this object. */
	/* Initializes this object. */
	void Initialize();
	/* Terminates this object. */
	/* Terminates this object. */
	void Terminate();

}