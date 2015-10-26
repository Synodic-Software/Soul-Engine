#pragma once

#include "Engine Core\Object\Object.cuh"

class ObjectSceneAbstraction
{
public:
	__host__ ObjectSceneAbstraction(Object*);
	__host__ ~ObjectSceneAbstraction();

	Object* object;
	ObjectSceneAbstraction* nextObject;
};

