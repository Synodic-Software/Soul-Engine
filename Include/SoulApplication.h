#pragma once
#include "Core/Soul.h"
#include "Core/SoulParameters.h"
#include "Display/Window/Window.h"

class SoulApplication {

public:

	SoulApplication(SoulParameters = SoulParameters());
	virtual ~SoulApplication() = default;

	std::shared_ptr<Window> CreateWindow(WindowParameters&);

	void Run();

protected:

	bool hasControl;
	SoulParameters parameters;

private:

	void CheckParameters();

	Soul soul;

};
