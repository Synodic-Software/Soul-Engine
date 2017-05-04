#pragma once

#include "BoundingBox.cuh"
#include "Ray Engine\CUDA/Ray.cuh"
#include "Engine Core\Object\Face.h"

class Node
{
public:
	__device__ __host__ Node();
	__device__ __host__ ~Node();


	Node* childLeft;
	Node* childRight;

	BoundingBox box;

	glm::mat4 transformRight;
	glm::mat4 transformLeft;

	/*union data{
		uint faceID;
		MiniObject* object;
	}data;*/

	uint faceID;
	uint atomic;
	uint rangeRight;
	uint rangeLeft;



	__host__ __device__ bool operator==(const Node& other) const {
		return
			childLeft == other.childLeft &&
			childRight == other.childRight &&
			rangeRight == other.rangeRight &&
			rangeLeft == other.rangeLeft&&
			transformLeft == other.transformLeft&&
			transformRight == other.transformRight;
	}

	__host__ __device__ Node& operator=(Node arg)
	{
		this->childLeft = arg.childLeft;
		this->childRight = arg.childRight;
		this->rangeRight = arg.rangeRight;
		this->rangeLeft = arg.rangeLeft;
		this->atomic = arg.atomic;
		this->faceID = arg.faceID;
		this->box = arg.box;
		this->transformLeft = arg.transformLeft;
		this->transformRight = arg.transformRight;

		return *this;
	}
	//__host__ __device__ void TransformRay(Ray&);

private:






};