#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"

enum castType{ RayCOLOUR, RayDISTANCE, RayOBJECT_ID, RayNORMAL, RayUV };
typedef void RayFunction();

class RayJob : public Managed{
public:

	//Takes an array of desired outputs, its size, the function that decides the ray generation, and the number of rays to shoot. 
	//The last parameter is the speed of the ray

	//some device function to be pointed to

	RayJob(castType, uint, RayFunction, uint);

private:

	castType type;
	uint typeSize;
	uint rayAmount;
	RayFunction raySetup;
};