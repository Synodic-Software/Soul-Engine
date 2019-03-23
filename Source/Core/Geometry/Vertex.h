#pragma once

#include "Types.h"
#include "Composition/Component/Component.h"

#include <glm/glm.hpp>
#include <functional>

class  Vertex : Component<Vertex>
{

public:

	Vertex() = default;
	~Vertex() = default;

	Vertex(const Vertex &) = default;
	Vertex(Vertex &&) noexcept = default;

	Vertex& operator=(const Vertex &) = default;
	Vertex& operator=(Vertex &&) noexcept = default;

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