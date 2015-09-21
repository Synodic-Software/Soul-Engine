#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"

enum castType{ RayCOLOUR, RayDISTANCE, RayOBJECT_ID, RayNORMAL, RayUV };
#define INFINITY 99999999999999.999999999999f

class RayJob : public Managed{
public:

	//Takes an array of desired outputs, its size, the function that decides the ray generation, and the number of rays to shoot. 
	//The last parameter is the speed of the ray
	RayJob(castType[], uint, ,uint, float);
private:

	castType* type;
	uint* rayAmount;
};