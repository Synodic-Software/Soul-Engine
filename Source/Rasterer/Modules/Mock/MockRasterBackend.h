#pragma once

#include "Rasterer/RasterModule.h"

class MockRasterBackend;

class MockRasterBackend final : public RasterModule {

public:

	MockRasterBackend() = default;
	~MockRasterBackend() override = default;

	MockRasterBackend(const MockRasterBackend &) = delete;
	MockRasterBackend(MockRasterBackend &&) noexcept = default;

	MockRasterBackend& operator=(const MockRasterBackend &) = delete;
	MockRasterBackend& operator=(MockRasterBackend &&) noexcept = default;

	void Draw() override;
	void DrawIndirect() override;


};
