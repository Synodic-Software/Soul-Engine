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
private:
	
};