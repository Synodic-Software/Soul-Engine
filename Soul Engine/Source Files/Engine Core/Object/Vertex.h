#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"

class Vertex : public Managed
{
public:
	Vertex();
	Vertex(float3, float2, float3);
	~Vertex();
private:
	float3* position;
	float2* textureCoord;
	float3* normal;
};