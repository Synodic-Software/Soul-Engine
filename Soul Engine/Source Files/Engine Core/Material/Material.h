#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Engine Core/Material/Texture/Texture.h"

static Texture* defaultTexture;

class Material : public Managed{
public:

	Material();
	~Material();


	void SetTexture(std::string);
	static void SetDefaultTexture(std::string);

private:
	//helper
	static cudaTextureObject_t defaultTex;

	cudaTextureObject_t tex;




	//general information
	std::string name;
	uint ID;


	//texture/mapping information
	bool hasTexture;

	glm::vec4 diffuse;
	std::string textureLocation;
	bool textureIsLoaded;

	bool hasDisplacement;

	Texture* displacement;


	//light information

	//physics information
};