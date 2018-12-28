#pragma once

#include "Rasterer/RasterBackendAPI.h"

class VulkanAPI : RasterBackendAPI {

public:

	~VulkanAPI() override = default;

	void Draw() override {}
	void DrawIndirect() override {}


};
