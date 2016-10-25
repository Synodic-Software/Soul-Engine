#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Engine Core\Object\Object.h"

//deal with hand model in soul engine
/*
this is now deprecated
from a point early in dev
objects were originally hardcoded in 
this is now *probably* unneeded
gonna leave it just in case
*/

class Hand: public Object
{
public:
	Hand(glm::vec3);
	~Hand();

	/*
	function declarations for hand.cpp
	*/
	void Update(double);
	void UpdateLate(double);
	void Load();

};

