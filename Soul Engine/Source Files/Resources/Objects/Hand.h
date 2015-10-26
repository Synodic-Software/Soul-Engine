#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Engine Core\Object\Object.cuh"

class Hand: public Object
{
public:
	Hand();
	~Hand();

	void Update(double);
	void UpdateLate(double);
	void Load();

};

