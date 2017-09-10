#pragma once

#include <cuda_runtime.h>

#include "Utility\Includes\GLMIncludes.h"
#include "Metrics.h"
#include <functional>

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

	uint object;

	__host__ __device__ bool operator==(const Vertex& other) const {
		return
			position == other.position &&
			normal == other.normal &&
			textureCoord == other.textureCoord &&
			velocity == other.velocity;
	}

	__host__ __device__ friend void swap(Vertex& a, Vertex& b)
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

		temp = a.velocity;
		a.velocity = b.velocity;
		b.velocity = temp;

		uint temp2 = a.object;
		a.object = b.object;
		b.object = temp2;

	}
	__host__ __device__ Vertex& operator=(Vertex arg)
	{
		this->position = arg.position;
		this->textureCoord = arg.textureCoord;
		this->normal = arg.normal;
		this->velocity = arg.velocity;

		return *this;
	}
private:

};

namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return (hash<float>()(vertex.position.x) ^
				hash<float>()(vertex.position.y) ^
				hash<float>()(vertex.position.z));

		}

	};

}