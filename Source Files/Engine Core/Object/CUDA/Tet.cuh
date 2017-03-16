#pragma once

#include "Engine Core/Material/Material.h"
#include "Engine Core/Object/CUDA/Object.cuh"
#include "Face.cuh"

class Object;

class Tet : public Managed
{
public:
	Tet();
	Tet(glm::uvec4, Material*);
	~Tet();


	glm::uvec4 indices;
	Material* materialPointer;
	Object* objectPointer;
private:
	
};
