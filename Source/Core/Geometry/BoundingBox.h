#pragma once

#include "Composition/Component/Component.h"

#include <glm/glm.hpp>

class BoundingBox : Component<BoundingBox>
{

public:

	BoundingBox() = default;
	~BoundingBox() = default;

	BoundingBox(const BoundingBox &) = default;
	BoundingBox(BoundingBox &&) noexcept = default;

	BoundingBox& operator=(const BoundingBox &) = default;
	BoundingBox& operator=(BoundingBox &&) noexcept = default;


	glm::vec3 min;
	glm::vec3 max;

};