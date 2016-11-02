#include "Vertex.cuh"


Vertex::Vertex()
{
	//position = glm::vec3(0.0f, 0.0f, 0.0f);
	//textureCoord = glm::vec2(0.0f, 0.0f);
	//normal = glm::vec3(0.0f, 0.0f, 0.0f);
	velocity = glm::vec3(0, 0, 0);
}
Vertex::Vertex(glm::vec3 posTemp, glm::vec2 uvTemp, glm::vec3 normTemp)
{
	position = posTemp;
	textureCoord = uvTemp;
	normal = normTemp;
	velocity = glm::vec3(0, 0, 0);
}

Vertex::~Vertex()
{
}
