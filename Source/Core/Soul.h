#pragma once

#include "SoulParameters.h"

class Soul {

public:

	Soul(SoulParameters&);

	void Initialize();
	void Terminate();

	void Run();

private:

	SoulParameters& parameters;
		
};
