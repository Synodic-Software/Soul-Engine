#pragma once
#include "Core/Soul.h"
#include "Core/SoulParameters.h"

class SoulApplication {

public:

	SoulApplication(SoulParameters = SoulParameters());
	virtual ~SoulApplication() = default;

	virtual void Initialize();
	virtual void Terminate();

	void Run();

protected:

	bool hasControl;
	SoulParameters parameters;

private:

	void CheckParameters();

	Soul soul;

};
