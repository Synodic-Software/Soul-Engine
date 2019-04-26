#pragma once

#include "Display/Window/WindowModule.h"

class MockWindowBackend final : public WindowModule {

public:

	MockWindowBackend(std::shared_ptr<InputModule>&);
	~MockWindowBackend() override = default;

	MockWindowBackend(const MockWindowBackend&) = delete;
	MockWindowBackend(MockWindowBackend&&) noexcept = default;

	MockWindowBackend& operator=(const MockWindowBackend&) = delete;
	MockWindowBackend& operator=(MockWindowBackend&&) noexcept = default;


	void Update() override;
	void Draw() override;
	bool Active() override;

	void CreateWindow(const WindowParameters&, std::shared_ptr<RasterModule>&) override;

	std::vector<const char*> GetRasterExtensions() override;

	Window& GetWindow() override;

};
