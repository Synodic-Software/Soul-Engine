#include "Photography/Film/CUDA/Film.cuh"

#include "cstdio"

Film::Film() 
:indicePointer(nullptr) {

}

Film::~Film() {

}

__device__ uint Film::GetIndex(uint x) {
	return x;
}

__device__ glm::vec2 Film::GetSample(uint in, curandState& randState) {
	uint id = GetIndex(in);

	glm::vec2 parameterization = glm::vec2(
		(curand_uniform(&randState) - 0.5f + id % resolution.x) / (resolution.x - 1),
		(curand_uniform(&randState) - 0.5f + id / resolution.x) / (resolution.y - 1)
	);

	return parameterization;
}
