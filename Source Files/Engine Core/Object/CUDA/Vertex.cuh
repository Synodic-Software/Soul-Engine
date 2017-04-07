#pragma once

#include "Utility\Includes\GLMIncludes.h"

class  Vertex
{
public:
	Vertex();
	Vertex(glm::vec3, glm::vec2, glm::vec3);
	~Vertex();

	glm::vec3 position;
	glm::vec2 textureCoord;
	glm::vec3 normal;

	glm::vec3 velocity;


	bool operator==(const Vertex& other) const {
		return position == other.position && normal == other.normal && textureCoord == other.textureCoord;
	}
	
	friend void swap(Vertex& a, Vertex& b)
	{

		glm::vec3 temp = a.position;
		a.position = b.position;
		b.position = temp;

		temp = a.normal;
		a.normal = b.normal;
		b.normal = temp;

		glm::vec2 temp1 = a.textureCoord;
		a.textureCoord = b.textureCoord;
		b.textureCoord = temp1;

	}
	Vertex& operator=(Vertex arg)
	{
		this->position = arg.position;
		this->textureCoord = arg.textureCoord;
		this->normal = arg.normal;
		this->velocity = arg.velocity;

		return *this;
	}
private:
	
};