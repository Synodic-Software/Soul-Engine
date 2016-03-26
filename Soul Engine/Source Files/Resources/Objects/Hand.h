#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Engine Core\Object\Object.h"

class Hand: public Object
{
public:
	Hand(glm::vec3);
	~Hand();

	void Update(double);
	void UpdateLate(double);
	void Load();

};

