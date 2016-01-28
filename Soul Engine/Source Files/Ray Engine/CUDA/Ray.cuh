#pragma once

#include "Utility\CUDAIncludes.h"

class Ray : public Managed{
public:

	//CUDA_FUNCTION Ray(glm::vec3 newO, glm::vec3 newD) : origin(newO), direction(newD){}
	CUDA_FUNCTION Ray(){
	}
	glm::vec3 origin;
	glm::vec3 direction;

};