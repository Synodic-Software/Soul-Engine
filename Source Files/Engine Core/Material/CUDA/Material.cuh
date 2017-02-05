#pragma once

#include "Engine Core/Material/Texture/Image.cuh"
#include "Utility\Includes\GLMIncludes.h"
#include <string>

class Material : public Managed{
public:

	__host__ Material(std::string texName = "SoulDefault.png");
	__host__ ~Material();

	glm::vec4 diffuse;
	glm::vec4 emit;

	Image image;

private:

};
