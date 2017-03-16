#pragma once

#include "Engine Core/Material/Material.h"
#include "Engine Core/Object/CUDA/Object.cuh"

class Object;
//__align__(32)
class Face : public Managed
{
public:
	Face();
	Face(glm::uvec3, Material*);
	~Face();

	void SetData(glm::uvec3, Material*);

	glm::uvec3 indices;
	Material* materialPointer;
	Object* objectPointer;
private:
	
};
