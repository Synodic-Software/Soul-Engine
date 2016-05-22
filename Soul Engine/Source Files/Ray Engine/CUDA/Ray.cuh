#pragma once

#include "Utility\CUDAIncludes.h"

//__align__(64)
class Ray : public Managed
{

public:

	//CUDA_FUNCTION Ray(glm::vec3 newO, glm::vec3 newD) : origin(newO), direction(newD){}
	CUDA_FUNCTION Ray();
	CUDA_FUNCTION Ray(const Ray &a);

	CUDA_FUNCTION Ray& Ray::operator= (const Ray &a)
	{

		origin = a.origin;
		direction = a.direction;
		storage = a.storage;
		job = a.job;
		resultOffset = a.resultOffset;
		active = a.active;

		return *this;
	}

	glm::vec4 storage;
	glm::vec3 origin; //origin and a single value representing the remaining distance this ray can travel in this frame
	uint resultOffset;
	glm::vec3 direction;
	char job;
	bool active;
	glm::vec4 padding;
	
private:


};