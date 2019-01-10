#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class RasterDevice : public Module {

public:

	virtual ~RasterDevice() = default;

	static std::shared_ptr<RasterDevice> CreateModule();

};
