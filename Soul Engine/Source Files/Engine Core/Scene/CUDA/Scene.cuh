#pragma once
#include <cuda.h>  

#include "Engine Core\Object\CUDA/Object.cuh"
#include "Engine Core\Scene\CUDA\Sky.cuh"
#include "Bounding Volume Heirarchy\BVH.h"

class Scene : public Managed
{
public:
	__host__ Scene();
	__host__ ~Scene();

	//adds all inthe queue and cleans all in the queue then builds the bvh
	__host__ void Build(float deltaTime);

	//signels the scene that an object should be added when the next 'Build()' is called
	//modifies the global scene bounding box, making the 3D spatial calculation less accurate
	__host__ uint AddObject(Object*);

	//signels the scene that an object should be removed when the next 'Build()' is called
	//Does NOT modify the global scene bounding box, meaning 3D spatial accuracy will remain as it was
	__host__ bool RemoveObject(const uint&);
BVH* bvh; 
Sky* sky;

private:
	//take in all requests for the frame and process them in bulk
	__host__ bool Scene::Compile();

	//abstraction layer that linearizes all seperate object pointer

	// a list of objects to remove 
	std::vector<uint> objectsToRemove;


	
	BoundingBox sceneBox;

	int newFaceAmount; //The amount of indices the entire scene takes
	int compiledSize; //the amount of indices as of the previous compile;
	int allocatedSize; //the amount of triangles that have been allocated

	bool* objectBitSetup; // hold a true for the first indice of each object
	uint* objIds; //points to the object
	Face** faceIds;
	uint64* mortonCodes;

	//Variables concerning object storage

	Object** objectList;
	bool* objectRemoval;
	uint objectsSize;
	uint allocatedObjects;

	
};

