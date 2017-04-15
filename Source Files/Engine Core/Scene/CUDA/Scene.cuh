#pragma once
#include <cuda.h>  

#include "Engine Core\Object\CUDA/Object.cuh"
#include "Engine Core\Object\CUDA/MiniObject.cuh"
#include "Engine Core\Scene\CUDA\Sky.cuh"
#include "Engine Core\Material\Material.h"
#include "Bounding Volume Heirarchy\BVH.h"
#include <vector>

class Scene
{
public:
	Scene();
	~Scene();

	//adds all inthe queue and cleans all in the queue then builds the bvh
	void Build(float deltaTime);

	//signels the scene that an object should be added when the next 'Build()' is called
	//modifies the global scene bounding box, making the 3D spatial calculation less accurate
	void AddObject(Object*);

	//signels the scene that an object should be removed when the next 'Build()' is called
	//Does NOT modify the global scene bounding box, meaning 3D spatial accuracy will remain as it was
	void RemoveObject(Object*);

	//the bvh data for the scene
	BVHData* bvhData;

	//the sky data for the scene
	Sky* sky;

	Face* faces;
	Vertex* vertices;
	Tet* tets;
	Material* materials;
	MiniObject* objects;

private:
	Sky* skyHost;

	BVH bvhHost;

	//updates the scene representation based on what is in addList or removeList
	void Compile();

	//scene bounding box
	BoundingBox sceneBox;

	uint64* mortonCodes; //codes for all the faces

	uint faceAmount;
	uint vertexAmount;
	uint tetAmount;
	uint materialAmount;
	uint objectAmount;

	uint faceAllocated;
	uint vertexAllocated;
	uint tetAllocated;
	uint materialAllocated;
	uint objectAllocated;

	std::vector<Object*> addList;
	std::vector<Object*> removeList;

};

