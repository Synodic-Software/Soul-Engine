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
	static const uint systemMin = 0;
	static const uint systemMax = 0xffffffff;

	BoundingBox* box;

	Node* childLeft;
	Node* childRight;

};