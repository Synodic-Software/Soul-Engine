#pragma once

#include "Core/Interface/Module/Module.h"

class RasterBackendAPI : public Module {

public:

	virtual ~RasterBackendAPI() = 0;

	virtual void Draw() = 0;
	virtual void DrawIndirect() = 0;


};
