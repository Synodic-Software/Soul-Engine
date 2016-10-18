#pragma once

#include "Utility\CUDA\CUDAManaged.cuh"
#include "Utility\GLMIncludes.h"

class  Vertex : public Managed
{
public:
	Vertex();
	Vertex(glm::vec3, glm::vec2, glm::vec3);
	~Vertex();

	void SetData(glm::vec3, glm::vec2, glm::vec3);

	glm::vec3 position;
	glm::vec2 textureCoord;
	glm::vec3 normal;

	glm::vec3 velocity;


	bool operator==(const Vertex& other) const {
		return position == other.position && normal == other.normal && textureCoord == other.textureCoord;
	}

	friend void swap(Vertex& a, Vertex& b)
	{
		using std::swap; // bring in swap for built-in types

		swap(a.position, b.position);
		swap(a.normal, b.normal);
		swap(a.textureCoord, b.textureCoord);
	}
private:
	
};