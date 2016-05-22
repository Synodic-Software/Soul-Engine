#include "Ray Engine\CUDA/Ray.cuh"

CUDA_FUNCTION Ray::Ray(){
}

CUDA_FUNCTION Ray::Ray(const Ray &a)
{
	origin = a.origin;
	direction = a.direction;
	storage = a.storage;
	job = a.job;
	resultOffset = a.resultOffset;
	active = a.active;
}