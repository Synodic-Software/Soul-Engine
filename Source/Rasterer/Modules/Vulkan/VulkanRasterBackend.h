#pragma once

#include "Rasterer/RasterBackend.h"

class VulkanRasterBackend : public RasterBackend {

public:

	~VulkanRasterBackend() override = default;

	void Draw() override {}
	void DrawIndirect() override {}


};