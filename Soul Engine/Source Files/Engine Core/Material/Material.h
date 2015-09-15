#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Engine Core/Material/Texture/Texture.h"

static Texture* defaultTexture;

class Material{
public:

	Material();
	~Material();

	GLuint64 GetHandle();
	void SetTexture(std::string);
	static void SetDefaultTexture(std::string);

private:
	//helper
	






	//general information
	std::string name;
	GLuint ID;


	//texture/mapping information
	bool hasTexture;

	glm::vec4 diffuse;
	std::string textureLocation;
	Texture* texture;
	GLuint64 textureHandle;
	bool textureIsLoaded;
	bool isResident;

	bool hasDisplacement;

	Texture* displacement;


	//light information

	//physics information
};