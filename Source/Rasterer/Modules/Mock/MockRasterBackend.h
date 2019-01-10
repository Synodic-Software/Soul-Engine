#pragma once

#include "Rasterer/RasterBackend.h"

class MockRasterBackend final: public RasterBackend {

public:

	~MockRasterBackend() override = default;

	void Draw() override {}
	void DrawIndirect() override {}


};