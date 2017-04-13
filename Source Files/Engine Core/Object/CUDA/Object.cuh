#pragma once

#include "Engine Core/Camera/CUDA/Camera.cuh"
#include "Input/Input.h"
#include "Engine Core/Material/Material.h"
#include "Engine Core/Object/CUDA/Vertex.cuh"
#include "Engine Core/Object/CUDA/Face.cuh"
#include "Engine Core/Object/CUDA/Tet.cuh"
#include "Bounding Volume Heirarchy\BoundingBox.h"
#include "Metrics.h"

#include <vector>

class Object {

public:

	Object();
	Object(std::string, Material*);
	~Object();

	bool requestRemoval;
	bool ready;
	bool isStatic;

	void AddVertices(Vertex*, uint);
	void AddFaces(Face*, uint);
	void ExtractFromFile(const char*);

	uint verticeAmount;
	uint faceAmount;
	uint tetAmount;
	uint materialAmount;

	std::vector<Vertex> vertices;
	std::vector<Face> faces;
	std::vector<Tet> tets;
	std::vector<Material> materials;

	void Update(double);
	void UpdateLate(double);
	void Load();

	BoundingBox box; //in object space

protected:

private:


};
