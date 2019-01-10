#pragma once

#include "Rasterer/RasterBackend.h"

class VulkanRasterBackend final: public RasterBackend {

public:

	~VulkanRasterBackend() override = default;

	void Draw() override {}
	void DrawIndirect() override {}


};