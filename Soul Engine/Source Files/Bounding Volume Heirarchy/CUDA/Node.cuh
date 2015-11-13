#pragma once

#include "Engine Core\BasicDependencies.h"
#include "BoundingBox.cuh"
#include "Ray Engine\Ray.cuh"

class Node : public Managed
{
public:
	Node();
	~Node();

	CUDA_FUNCTION void TransformRay(Ray&);

private:
	float systemMin = 0.0f;
	float systemMax = 8192.0f;

	BoundingBox* box;

	Node* childLeft;
	Node* childRight;

};