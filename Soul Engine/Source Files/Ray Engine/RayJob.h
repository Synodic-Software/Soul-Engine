#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"

enum castType{ RayCOLOUR, RayDISTANCE, RayOBJECT_ID, RayNORMAL, RayUV };

class RayJob : public Managed{
public:

	//Takes an array of desired outputs, its size, the function that decides the ray generation, and the number of rays to shoot.
	RayJob(castType[], uint, ,uint);
private:

	castType* type;
	uint* rayAmount;
};