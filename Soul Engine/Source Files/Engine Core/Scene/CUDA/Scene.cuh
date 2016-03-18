#pragma once

#include "Utility\CUDAIncludes.h"
#include "Engine Core\Object\CUDA/Object.cuh"
#include "ObjectSceneAbstraction.cuh"
#include "Ray Engine\CUDA/Ray.cuh"
#include "Bounding Volume Heirarchy\CUDA\Node.cuh"
#include <thrust/fill.h>
#include "Algorithms\Data Algorithms\GPU Prefix Sum\PrefixSum.h"
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/functional.h>



class Scene : public Managed
{
public:
	__host__ Scene();
	__host__ ~Scene();

	__device__ glm::vec3 IntersectColour(Ray& ray, curandState&)const;

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
	bool* objectBitSetup; // hold a true for the first indice of each object

	uint* objIds; //points to the object

	//for object storage
	Object* objectList;
	uint objectsSize;
	uint maxObjects;

};

