#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Ray Engine\Ray.cuh"
#include "Engine Core\Camera\CUDA\Camera.cuh"

enum castType{ RayCOLOUR, RayCOLOUR_TO_BUFFER, RayDISTANCE, RayOBJECT_ID, RayNORMAL, RayUV };

class RayJob : public Managed{
public:

	//Takes an array of desired outputs, its size, the function that decides the ray generation, and the number of rays to shoot. 
	//The last parameter is the speed of the ray

	//some device function to be pointed to

	__host__ RayJob(castType, uint, uint, Camera* camera, bool isRecurring);
	__host__ ~RayJob();
	RayJob* nextRay;	

	Camera* camera;

	CUDA_FUNCTION bool IsRecurring() const{
		return isRecurring;
	}
	CUDA_FUNCTION void ChangeProbability(float);
	CUDA_FUNCTION float GetProbability();

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
private:

	bool isRecurring;
	float probability;
	//result containers

	//for texture setup
	

	
};