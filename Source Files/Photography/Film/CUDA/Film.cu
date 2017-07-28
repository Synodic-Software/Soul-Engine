#include "Photography/Film/CUDA/Film.cuh"

Film::Film() {

}

Film::~Film() {

}


__device__ glm::vec2 Film::GetSample(uint id, curandState& randState) {

	glm::vec2 parameterization = glm::vec2(
		(curand_uniform(&randState) - 0.5f + id % resolution.x) / (resolution.x - 1),
		(curand_uniform(&randState) - 0.5f + id / resolution.x) / (resolution.y - 1)
	);

	return parameterization;
}
