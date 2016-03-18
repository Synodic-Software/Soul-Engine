#pragma once

#include "Utility\CUDAIncludes.h"

class Ray : public Managed{
public:

	//CUDA_FUNCTION Ray(glm::vec3 newO, glm::vec3 newD) : origin(newO), direction(newD){}
	CUDA_FUNCTION Ray(){
	}

	CUDA_FUNCTION Ray(const Ray &a)
	{
		origin = a.origin;
		direction = a.direction;
		storage = a.storage;
		job = a.job;
		resultOffset = a.resultOffset;
		active = a.active;
	}

	CUDA_FUNCTION Ray& Ray::operator= (const Ray &a)
	{
		//// check for self-assignment by comparing the address of the
		//// implicit object and the parameter
		//if (this == &a)
		//	return *this;

		// do the copy
		origin = a.origin;
		direction = a.direction;
		storage = a.storage;
		job = a.job;
		resultOffset = a.resultOffset;
		active = a.active;
		// return the existing object
		return *this;
	}

	glm::vec3 origin;
	glm::vec3 direction;
	glm::vec4 storage;
	uint resultOffset;
	bool active;
	char job;
};