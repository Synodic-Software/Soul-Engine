#pragma once

#include "Engine Core/Material/Texture/Image.cuh"
#include "Utility\Includes\GLMIncludes.h"
#include <string>

class Material {
public:

	Material(std::string texName = "Resources\\Textures\\SoulDefault.png");
	~Material();

	glm::vec4 diffuse;
	glm::vec4 emit;

	//Image diffuseImage;

	__host__ __device__ bool operator==(const Material& other) const {
		return
			diffuse == other.diffuse &&
			emit == other.emit; //&&
			//diffuseImage == other.diffuseImage;
	}

	__host__ __device__ friend void swap(Material& a, Material& b)
	{

		glm::vec4 temp = a.diffuse;
		a.diffuse = b.diffuse;
		b.diffuse = temp;

		temp = a.emit;
		a.emit = b.emit;
		b.emit = temp;

		//swap(a.diffuseImage, b.diffuseImage);
	}
	__host__ __device__ Material& operator=(Material arg)
	{
		this->diffuse = arg.diffuse;
		this->emit = arg.emit;
		//this->diffuseImage = arg.diffuseImage;

		return *this;
	}
private:

};
