#pragma once

#include "Utility\Includes\GLMIncludes.h"
#include "Metrics.h"
#include <functional>

class  Vertex
{

public:

	bool operator==(const Vertex& other) const;

	glm::vec3 position;
	glm::vec2 textureCoord;
	glm::vec3 normal;
	glm::vec3 velocity;

	uint object;

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