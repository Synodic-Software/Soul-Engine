#pragma once

#include "Display/Display.h"

class MockDisplay final: public Display {

public:

	~MockDisplay() override = default;

	MockDisplay(const MockDisplay&) = delete;
	MockDisplay(MockDisplay&&) noexcept = default;

	MockDisplay& operator=(const MockDisplay&) = delete;
	MockDisplay& operator=(MockDisplay&&) noexcept = default;

	void Draw() override;
	bool Active() override;

	void CreateWindow(const WindowParameters&, RasterBackend*) override;
	void RegisterRasterBackend(RasterBackend*) override;

};
