#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Ray Engine\Ray.cuh"
#include "Engine Core\Camera\CUDA\Camera.cuh"


enum castType{ 
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
	//@param The number of samples per ray or point that will be averaged into the result
	//@param A camera that contains all the information to shoot a ray.
	//@param Boolean inquiring whether or not to use memory optimizations if a job is per-frame.
	__host__ RayJob(castType, uint, uint, Camera* camera, bool isRecurring);
	__host__ ~RayJob();


	CUDA_FUNCTION Camera* GetCamera(){
		return camera;
	}
		
	CUDA_FUNCTION bool IsRecurring() const{
		return isRecurring;
	}





	CUDA_FUNCTION void ChangeProbability(float);
	CUDA_FUNCTION float SampleProbability();


	uint samples;
	castType type;
	uint rayAmount;
	uint rayBaseAmount;

//for texture setup
	glm::vec4* resultsT;

	//for float values
	glm::vec3* resultsF;

	//for int values
	uint* resultsI;

protected:

private:

	bool isRecurring;
	float probability;
	Camera* camera;

	//result containers

	//for texture setup
	

	
};