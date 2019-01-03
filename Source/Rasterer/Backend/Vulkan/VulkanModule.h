#pragma once

#include "Rasterer/RasterBackendAPI.h"

class VulkanRasterModule : public RasterBackendAPI {

public:

	~VulkanRasterModule() override = default;

	void Draw() override {}
	void DrawIndirect() override {}


};