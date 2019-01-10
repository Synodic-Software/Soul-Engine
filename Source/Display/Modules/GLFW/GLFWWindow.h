#pragma once

#include "Display/Window.h"

#include <GLFW/glfw3.h>

class GLFWWindow final : public Window
{

public:

	GLFWWindow(WindowParameters&, GLFWmonitor*);
	~GLFWWindow() override;

	GLFWWindow(const GLFWWindow &) = delete;
	GLFWWindow(GLFWWindow &&) noexcept = default;

	GLFWWindow& operator=(const GLFWWindow &) = delete;
	GLFWWindow& operator=(GLFWWindow &&) noexcept = default;

	void Refresh();
	void Close();
	void Resize(int, int);
	void PositionUpdate(int, int);
	void FrameBufferResize(int, int);

	GLFWwindow* Context();


	private:

		GLFWwindow* context_;

};
