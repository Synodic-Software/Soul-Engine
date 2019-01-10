#pragma once

#include "Rasterer/RasterBackend.h"

class MockRasterBackend : public RasterBackend {

public:

	~MockRasterBackend() override = default;

	void Draw() override {}
	void DrawIndirect() override {}


};