#pragma once

#include "Engine Core\Object\Object.h"
#include "Engine Core\Object\MiniObject.h"
#include "Engine Core\Scene\Sky.h"
#include "Engine Core\Material\Material.h"
#include "Data Structures\Bounding Volume Heirarchy\BVH.h"

#include "Compute/ComputeBuffer.h"

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
	ComputeBuffer<BVHData> bvhData;

	//the sky data for the scene
	ComputeBuffer<Sky> sky;

	ComputeBuffer<Face> faces;
	ComputeBuffer<Vertex> vertices;
	ComputeBuffer<Tet> tets;
	ComputeBuffer<Material> materials;
	ComputeBuffer<MiniObject> objects;

private:

	//updates the scene representation based on what is in addList or removeList
	void Compile();

	//scene bounding box
	BoundingBox sceneBox;

	ComputeBuffer<BVH> bvh;
	ComputeBuffer<uint64> mortonCodes; //codes for all the faces

};

