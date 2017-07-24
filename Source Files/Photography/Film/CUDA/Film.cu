#include "Photography/Film/CUDA/Film.cuh"

Film::Film() {

}

Film::~Film() {

}

uint Film::GetIndex(uint x) {
	return x;
}

glm::vec2 Film::GetNormalized(uint) {

	return glm::vec2(0.5f);
}
