#pragma once

#include "Rasterer/RasterBackendAPI.h"

class MockRasterModule : public RasterBackendAPI {

public:

	~MockRasterModule() override = default;

	void Draw() override {}
	void DrawIndirect() override {}


};