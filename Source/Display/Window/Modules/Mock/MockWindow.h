#pragma once

#include "Display/Window/Window.h"

class MockWindow : public Window {

public:

	MockWindow();
	~MockWindow() override = default;

	MockWindow(const MockWindow&) = delete;
	MockWindow(MockWindow&&) noexcept = default;

	MockWindow& operator=(const MockWindow&) = delete;
	MockWindow& operator=(MockWindow&&) noexcept = default;


};
