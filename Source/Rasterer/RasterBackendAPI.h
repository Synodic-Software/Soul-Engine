#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class RasterBackendAPI : public Module {

public:

	virtual ~RasterBackendAPI() = default;

	virtual void Draw() = 0;
	virtual void DrawIndirect() = 0;

	static std::shared_ptr<RasterBackendAPI> CreateModule();

};
