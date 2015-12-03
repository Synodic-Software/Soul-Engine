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
	uint systemMin;
	uint systemMax;

	BoundingBox* box;

	Node* childLeft;
	Node* childRight;

};