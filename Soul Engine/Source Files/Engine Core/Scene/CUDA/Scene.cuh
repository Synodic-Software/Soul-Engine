#pragma once

#include "Utility\CUDAIncludes.h"
#include "Engine Core\Object\CUDA/Object.cuh"
#include "ObjectSceneAbstraction.cuh"
#include "Ray Engine\CUDA/Ray.cuh"
#include "Bounding Volume Heirarchy\CUDA\Node.cuh"

class Scene : public Managed
{
public:
	__host__ Scene();
	__host__ ~Scene();

	CUDA_FUNCTION glm::vec3 IntersectColour(const Ray& ray)const;

	//adds all inthe queue and cleans all in the queue then builds the bvh
	__host__ void Build();
	__host__ void AddObject(Object&);
	__host__ bool Scene::Clean();
	__host__ bool Scene::Compile();
private:
	__host__ void AttachObjIds();
	//abstraction layer that linearizes all seperate object pointer
	Node* BVH; 

	uint indicesSize; //The amount of indices the entire scene takes
	bool* objectBitSetup;

	

	//for object storage
	Object* objectList;
	uint objectsSize;
	uint maxObjects;

};

