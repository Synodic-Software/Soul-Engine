#pragma once

#include "Types.h"
#include "Composition/Component/Component.h"

#include <glm/glm.hpp>


class Vertex : Component<Vertex>
{

public:

	Vertex() = default;
	~Vertex() = default;

	Vertex(const Vertex&) = default;
	Vertex(Vertex&&) noexcept = default;

	Vertex& operator=(const Vertex&) = default;
	Vertex& operator=(Vertex&&) noexcept = default;


	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 textureCoord;
	glm::vec3 velocity;

	uint object;

};