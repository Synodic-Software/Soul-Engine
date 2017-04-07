#pragma once

#include "Engine Core/Camera/CUDA/Camera.cuh"
#include "Input/Input.h"
#include "Engine Core/Material/Material.h"
#include "Engine Core/Object/CUDA/Vertex.cuh"
#include "Engine Core/Object/CUDA/Face.cuh"
#include "Bounding Volume Heirarchy\BoundingBox.h"
#include "Metrics.h"

class Face;


namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<float>()(vertex.position.x) ^
				hash<float>()(vertex.position.y) ^
				hash<float>()(vertex.position.z) ^
				(hash<float>()(vertex.normal.x) << 1) ^
				(hash<float>()(vertex.normal.y) << 1) ^
				(hash<float>()(vertex.normal.z) << 1)) >> 1) ^
				(hash<float>()(vertex.textureCoord.x) << 1) ^
				(hash<float>()(vertex.textureCoord.y) << 1);
		}
	};
}

class Object{
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
		Vertex* verticesCPU;
		Face* faces;
		Face* facesCPU;

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
