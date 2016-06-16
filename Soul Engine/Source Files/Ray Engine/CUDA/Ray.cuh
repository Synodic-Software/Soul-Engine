#pragma once

#include "Utility\CUDAIncludes.h"
#include "Engine Core\Object\CUDA\Face.cuh"

class Face;
//__align__(64)
class Ray : public Managed
{

public:

	//CUDA_FUNCTION Ray(glm::vec3 newO, glm::vec3 newD) : origin(newO), direction(newD){}
	CUDA_FUNCTION Ray();
	CUDA_FUNCTION Ray::Ray(const Ray &a)
	{
		storage = a.storage;
		origin = a.origin;
		direction = a.direction;
		bary = a.bary;
		currentHit = a.currentHit;
		resultOffset = a.resultOffset;
		job = a.job;
	}

	CUDA_FUNCTION Ray& Ray::operator= (const Ray &a)
	{

		storage = a.storage;
		origin = a.origin;
		direction = a.direction;
		bary = a.bary;
		currentHit = a.currentHit;
		resultOffset = a.resultOffset;
		job = a.job;

		return *this;
	}

	glm::vec4 storage;
	glm::vec4 origin; //origin and a single value representing the remaining distance this ray can travel in this frame
	glm::vec4 direction;
	glm::vec2 bary;
	Face* currentHit;
	uint resultOffset;
	char job;
	
private:


};