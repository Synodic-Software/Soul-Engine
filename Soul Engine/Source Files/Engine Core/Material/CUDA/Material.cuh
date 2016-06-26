#pragma once
#include "Utility\CUDAIncludes.h"
//#include "Engine Core/BasicDependencies.h"
#include "Utility/OpenGL/ShaderSupport.h"
//#include "Engine Core/Material/Texture/Texture.h"
#include "Engine Core/Material/Texture/Image.cuh"


class Material : public Managed{
public:

	__host__ Material(std::string texName = "SoulDefault.png");
	__host__ ~Material();


	//void SetTexture(std::string);
	//static void SetDefaultTexture(std::string);
	glm::vec4 diffuse;
	glm::vec4 emit;

	Image image;

private:

};
