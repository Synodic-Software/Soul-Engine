#pragma once

#include "Engine Core/Camera/CUDA/Camera.cuh"
#include "Input/Input.h"
#include "Engine Core/Material/Material.h"
#include "Engine Core/Object/CUDA/Vertex.cuh"
#include "Engine Core/Object/CUDA/Face.cuh"
#include "Bounding Volume Heirarchy\BoundingBox.h"
#include "Metrics.h"

class Face;

class Object: public Managed{
	public:

		Object();

		Object(glm::vec3,std::string,Material*);

		bool requestRemoval;
		bool ready;
		bool isStatic;

		glm::vec3 xyzPosition;
		glm::vec3 velocity;
		glm::vec3 acceleration;

		void AddVertices(Vertex*,uint);
		void AddFaces(Face*, uint);
		void ExtractFromFile(const char*);

		uint verticeAmount;
		uint faceAmount;

		Vertex* vertices;
		Face* faces;

	void Update(double);
	void UpdateLate(double);
	void Load();

	Material** materialP;
	uint materialSize;

	BoundingBox box;

	uint localSceneIndex;
protected:
	
private:

	
};
