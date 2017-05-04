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
	Node* parent;

	BoundingBox box;

	glm::mat4 transform;

	uint64 morton;
	uint bestObjID;
	uint bitOffset;
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
			transform == other.transform&&
			morton == other.morton&&
			bestObjID == other.bestObjID&&
			bitOffset == other.bitOffset;

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
		this->transform = arg.transform;
		this->morton = arg.morton;
		this->bestObjID = arg.bestObjID;
		this->bitOffset = arg.bitOffset;

		return *this;
	}
	//__host__ __device__ void TransformRay(Ray&);

private:






};