#pragma once

#include "BoundingBox.cuh"
#include "Ray Engine\CUDA/Ray.cuh"
#include "Engine Core\Object\Face.h"

class Node
{

public:

	__device__ __host__ Node();

	uint childLeft;
	uint childRight;

	BoundingBox box;

	uint faceID;
	uint atomic;
	uint rangeRight;
	uint rangeLeft;

};