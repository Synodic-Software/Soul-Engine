#pragma once

#include "Display/Display.h"

class GLFWDisplay : public Display {

public:

	GLFWDisplay() = default;
	~GLFWDisplay() override = default;

	void Draw() override;
	std::shared_ptr<Window> CreateWindow(WindowParameters&) override;
	bool ShouldClose() override;

};
