#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"



class Ray : public Managed{
public:

	Ray(float3 newO, float3 newD) : origin(newO), direction(newD){}
	float3 origin;
	float3 direction;

};