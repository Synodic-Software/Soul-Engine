#pragma once

#include "Utility\CUDAIncludes.h"
#include "Ray Engine\CUDA/Ray.cuh"
#include "Engine Core\Camera\CUDA/Camera.cuh"


enum rayType{ 
	  RayCOLOUR				//RayCOLOUR: A vec3 of RGB values to be displayed
	, RayCOLOUR_SPECTRAL	//RayCOLOUR_SPECTRAL: Uses additional processing time to perform spectral operations and returns the result in vec3 RGB space.
	, RayDISTANCE			//RayDISTANCE: A float of the specific ray's distance travelled.
	, RayOBJECT_ID			//RayOBJECT_ID: A unique ID of the first Object hit in a uint.
	, RayNORMAL				//RayNORMAL: The normal at the first point hit in a vec3.
	, RayUV					//RayUV: The UV at the first point hit in a vec2.
};

class RayJob : public Managed{
public:

	//@param The information to be retreiving from the job.
	//@param The number of rays/data-points to be cast into the scene.
	//@param The number of samples per ray or point that will be averaged into the result. Is more of a probability than number.
	//@param A camera that contains all the information to shoot a ray.
	//@param Boolean inquiring whether or not to use memory optimizations if a job is per-frame.
	__host__ RayJob(rayType, uint, float, Camera* camera, bool isRecurring);
	__host__ ~RayJob();


	//Returns a reference to a camera pointer. All the ray shooting information is stored here.
	CUDA_FUNCTION Camera*& GetCamera();
		
	//Returns a boolean of the jobs storage flag.
	CUDA_FUNCTION bool IsRecurring() const;

	//Returns the rayType of the job.
	CUDA_FUNCTION rayType RayType() const;

	//Returns the Ray max of the job as per its initialization params.
	CUDA_FUNCTION uint RayAmountMax() const;

	//Returns the current rayAmount (modifiable)
	CUDA_FUNCTION uint& GetRayAmount();

	//Returns the current sample per ray (modifiable)
	CUDA_FUNCTION float& GetSampleAmount();


	//Returns the pointer to the results (modifiable)
	CUDA_FUNCTION void*& GetResultPointer();
	

protected:

private:

	//counting variables
	uint startIndex;

	//common variables
	bool isRecurring;
	Camera* camera;
    rayType type;
	uint rayAmount;
	uint rayBaseAmount;
	float samples;

	//result variables
	void* results;
};