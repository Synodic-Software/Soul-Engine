#pragma once
#include <cuda.h>  

#include "Engine Core\Object\Object.h"
#include "Engine Core\Object\MiniObject.h"
#include "Engine Core\Scene\Sky.h"
#include "Engine Core\Material\Material.h"
#include "Data Structures\Bounding Volume Heirarchy\BVH.h"

#include "GPGPU/GPUBuffer.h"

class Scene
{
public:
	Scene();

	~Scene() = default;

	//adds all inthe queue and cleans all in the queue then builds the bvh
	void Build(float deltaTime);

	//signels the scene that an object should be added when the next 'Build()' is called
	//modifies the global scene bounding box, making the 3D spatial calculation less accurate
	void AddObject(Object&);

	//signels the scene that an object should be removed when the next 'Build()' is called
	//Does NOT modify the global scene bounding box, meaning 3D spatial accuracy will remain as it was
	void RemoveObject(Object&);

	//the bvh data for the scene
	GPUBuffer<BVHData> bvhData;

	//the sky data for the scene
	GPUBuffer<Sky> sky;

	GPUBuffer<Face> faces;
	GPUBuffer<Vertex> vertices;
	GPUBuffer<Tet> tets;
	GPUBuffer<Material> materials;
	GPUBuffer<MiniObject> objects;

private:

	//updates the scene representation based on what is in addList or removeList
	void Compile();

	//scene bounding box
	BoundingBox sceneBox;

	GPUBuffer<BVH> bvh;
	GPUBuffer<uint64> mortonCodes; //codes for all the faces

};

