#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"
#include "Ray Engine\Ray.cuh"
#include "Engine Core\Camera\CUDA\Camera.h"

enum castType{ RayCOLOUR, RayCOLOUR_TO_TEXTURE, RayDISTANCE, RayOBJECT_ID, RayNORMAL, RayUV };

class RayJob : public Managed{
public:

	//Takes an array of desired outputs, its size, the function that decides the ray generation, and the number of rays to shoot. 
	//The last parameter is the speed of the ray

	//some device function to be pointed to

	RayJob(castType, uint, uint, Camera* camera, bool isRecurring);

	RayJob* nextRay;	

	Camera* camera;

	CUDA_FUNCTION bool IsRecurring() const{
		return isRecurring;
	}
	CUDA_FUNCTION glm::vec4* GetResultBuffer(){
		return resultsT;
	}
	CUDA_FUNCTION glm::vec3* GetResultFloat(){
		return resultsF;
	}
	CUDA_FUNCTION uint* GetResultInt(){
		return resultsI;
	}

	uint samples;
	castType type;
	uint rayAmount;
	uint rayBaseAmount;

private:

	bool isRecurring;

	//result containers

	//for texture setup
	

	//for texture setup
	glm::vec4* resultsT;

	//for float values
	glm::vec3* resultsF;

	//for int values
	uint* resultsI;
};