#pragma once

#include "Display/Display.h"

#include "Display/Modules/GLFW/GLFWWindow.h"

#include <vector>

struct GLFWmonitor;
class WindowParameters;

class GLFWDisplay final : public Display {

public:

	GLFWDisplay();
	~GLFWDisplay() override;

	GLFWDisplay(const GLFWDisplay&) = delete;
	GLFWDisplay(GLFWDisplay&&) noexcept = default;

	GLFWDisplay& operator=(const GLFWDisplay&) = delete;
	GLFWDisplay& operator=(GLFWDisplay&&) noexcept = default;

	void Draw() override;
	bool ShouldClose() override;

	std::shared_ptr<Window> CreateWindow(WindowParameters&) override;


private:

	std::shared_ptr<GLFWWindow> masterWindow_;

	std::vector<GLFWmonitor*> monitors_;
	std::vector<std::shared_ptr<GLFWWindow>> windows_;

};
