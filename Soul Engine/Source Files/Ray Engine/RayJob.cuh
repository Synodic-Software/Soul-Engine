#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"
#include "Ray Engine\Ray.cuh"
enum castType{ RayCOLOUR, RayDISTANCE, RayOBJECT_ID, RayNORMAL, RayUV };

//forward,right, origin, distance from camera, fov, index
typedef Ray(*RayFunction)(RayJob&, uint);

#define RAY_FUNCTION __device__ RayFunction

class RayJob : public Managed{
public:

	//Takes an array of desired outputs, its size, the function that decides the ray generation, and the number of rays to shoot. 
	//The last parameter is the speed of the ray

	//some device function to be pointed to

	RayJob(castType, RayFunction, uint, uint, bool, float3, float3, float3, float, float2);
	bool IsReaccuring() const{
		return isReaccuring;
	}
	RayJob* nextRay=NULL;	

	float3 forward;
	float3 right;
	float3 origin;
	float distanceFromO;
	float2 fov;

	float3* resultsF;
	uint1* resultsI;


	uint samples;
	castType type;
	uint rayAmount;
	uint rayBaseAmount;
	RayFunction raySetup;
private:
	bool isReaccuring;

};