#pragma once

#include "Rasterer/RasterBackendAPI.h"

class MockAPI : RasterBackendAPI {

public:

	~MockAPI() override = default;

	void Draw() override {}
	void DrawIndirect() override {}


};
