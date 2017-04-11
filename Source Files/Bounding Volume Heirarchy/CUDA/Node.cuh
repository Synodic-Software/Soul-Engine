#pragma once

#include "BoundingBox.cuh"
#include "Ray Engine\CUDA/Ray.cuh"
#include "Engine Core\Object\Face.h"

class Node
{
public:
	Node();
	~Node();


	Node* childLeft;
	Node* childRight;

	BoundingBox box;

	
	uint faceID;
	uint atomic;
	uint rangeRight;
	uint rangeLeft;


	//__host__ __device__ void TransformRay(Ray&);

private:


	

	

};