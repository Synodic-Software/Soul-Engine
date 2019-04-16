#pragma once

#include "Display/DisplayModule.h"

class MockDisplay final : public DisplayModule {

public:

	~MockDisplay() override = default;

	MockDisplay(const MockDisplay&) = delete;
	MockDisplay(MockDisplay&&) noexcept = default;

	MockDisplay& operator=(const MockDisplay&) = delete;
	MockDisplay& operator=(MockDisplay&&) noexcept = default;

	void Draw() override;
	bool Active() override;

	void CreateWindow(const WindowParameters&, RasterModule*) override;
	void RegisterRasterBackend(RasterModule*) override;

};
