#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class RasterBackend : public Module {

public:

	virtual ~RasterBackend() = default;

	virtual void Draw() = 0;
	virtual void DrawIndirect() = 0;

	static std::shared_ptr<RasterBackend> CreateModule();

};
