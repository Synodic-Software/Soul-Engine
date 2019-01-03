#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class RasterDeviceAPI : public Module {

public:

	virtual ~RasterDeviceAPI() = default;

	static std::shared_ptr<RasterDeviceAPI> CreateModule();

};
