#pragma once

#include "Display/Window.h"

struct GLFWmonitor;
struct GLFWwindow;
class RasterBackend;

class GLFWWindow final : public Window
{

public:

	GLFWWindow(const WindowParameters&, GLFWmonitor*, RasterBackend*, bool);
	~GLFWWindow() override;

	GLFWWindow(const GLFWWindow &) = delete;
	GLFWWindow(GLFWWindow &&) noexcept = default;

	GLFWWindow& operator=(const GLFWWindow &) = delete;
	GLFWWindow& operator=(GLFWWindow &&) noexcept = default;


	GLFWwindow* Context() const;
	bool Master() const;

private:

	GLFWwindow* context_;

	bool master_;

};
