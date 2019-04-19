#pragma once

#include "Display/Window/WindowModule.h"

class MockWindowBackend final : public WindowModule {

public:

	~MockWindowBackend() override = default;

	MockWindowBackend(const MockWindowBackend&) = delete;
	MockWindowBackend(MockWindowBackend&&) noexcept = default;

	MockWindowBackend& operator=(const MockWindowBackend&) = delete;
	MockWindowBackend& operator=(MockWindowBackend&&) noexcept = default;

	void Draw() override;
	bool Active() override;

	void CreateWindow(const WindowParameters&, RasterModule*) override;
	void RegisterRasterBackend(RasterModule*) override;

};
