#pragma once

#include "Core/Composition/Component/Component.h"

#include "Types.h"
#include <glm/glm.hpp>

class Tet : Component<Tet>
{

public:

	Tet() = default;
	~Tet() = default;

	Tet(const Tet &) = default;
	Tet(Tet &&) noexcept = default;

	Tet& operator=(const Tet &) = default;
	Tet& operator=(Tet &&) noexcept = default;

	glm::uvec4 indices;
	uint material;
	uint object;

};
