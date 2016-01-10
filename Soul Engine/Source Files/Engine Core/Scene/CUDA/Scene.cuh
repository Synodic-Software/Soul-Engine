#pragma once

#include "Utility\CUDAIncludes.h"
#include "Engine Core\Object\CUDA/Object.cuh"
#include "ObjectSceneAbstraction.cuh"
#include "Ray Engine\CUDA/Ray.cuh"

class Scene : public Managed
{
public:
	__host__ Scene();
	__host__ ~Scene();

	CUDA_FUNCTION glm::vec3 IntersectColour(const Ray& ray)const;
	__host__ void AddObject(Object&);

private:

	uint* mortonCodes;
	uint* objectPointers;

	uint indicesSize;


	Object* objectList;
	uint objectsSize;
	uint maxObjects;

};

