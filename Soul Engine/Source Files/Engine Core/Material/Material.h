#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Engine Core/Material/Texture/Texture.h"
#include "Utility\GPUIncludes.h"

static Texture* defaultTexture;

class Material : public Managed{
public:

	Material();
	~Material();


	void SetTexture(std::string);
	static void SetDefaultTexture(std::string);


	glm::vec4 diffuse;
	glm::vec4 emit;

private:
	//helper
	static cudaTextureObject_t defaultTex;

	cudaTextureObject_t tex;




	//general information
	std::string name;
	uint ID;


	//texture/mapping information
	bool hasTexture;

	std::string textureLocation;
	bool textureIsLoaded;

	bool hasDisplacement;

	Texture* displacement;


	//light information

	//physics information
};