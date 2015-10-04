#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\CUDAManaged.cu"



class Ray : public Managed{
public:

	Ray(glm::vec3 newO, glm::vec3 newD) : origin(newO), direction(newD){}
	glm::vec3 origin;
	glm::vec3 direction;

};