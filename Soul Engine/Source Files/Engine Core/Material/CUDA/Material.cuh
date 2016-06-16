#pragma once
#include "Utility\CUDAIncludes.h"
//#include "Engine Core/BasicDependencies.h"
#include "Utility/OpenGL/ShaderSupport.h"
//#include "Engine Core/Material/Texture/Texture.h"
#include "Engine Core/Material/Texture/Image.h"


class Material : public Managed{
public:

	__host__ Material(std::string texName = "SoulDefault.png");
	__host__ ~Material();


	//void SetTexture(std::string);
	//static void SetDefaultTexture(std::string);
	cudaTextureObject_t texObj;

	glm::vec4 diffuse;
	glm::vec4 emit;
private:
	Image image;

};
