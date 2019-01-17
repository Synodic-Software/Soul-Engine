#pragma once
#include "Core/Soul.h"
#include "Core/SoulParameters.h"
#include "Display/Window.h"

class SoulApplication {

public:

	SoulApplication(SoulParameters = SoulParameters());
	virtual ~SoulApplication() = default;

	void CreateWindow(WindowParameters&);

	void Run();

protected:

	bool hasControl;
	SoulParameters parameters;

private:

	void CheckParameters();

	Soul soul;

};
