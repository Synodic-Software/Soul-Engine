#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"

enum castType{ RayCOLOUR, RayDISTANCE, RayOBJECT_ID, RayNORMAL, RayUV };

class RayJob : public Managed{
public:
	RayJob();
	RayJob(castType, uint);
private:
	castType* type;
	uint* rayAmount;
};