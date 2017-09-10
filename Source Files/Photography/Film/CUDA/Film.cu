#include "Photography/Film/CUDA/Film.cuh"
#include "stdio.h"

Film::Film() {

}

Film::~Film() {

}


__device__ glm::vec2 Film::GetSample(uint id, curandState& randState) {

	float xMax = resolution.x;
	float yMax = resolution.y;

	glm::vec2 parameterization = glm::vec2(
		(curand_uniform(&randState) - 0.5f + id % resolution.x) / (xMax - 1),
		(curand_uniform(&randState) - 0.5f + id / xMax) / (yMax - 1)
	);

	return parameterization;
}
