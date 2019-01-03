#pragma once

#include "Composition/Component/Component.h"

class GLFWWindow : Component<GLFWWindow>
{
public:

	GLFWWindow() = default;
	~GLFWWindow() = default;

	GLFWWindow(const GLFWWindow &) = delete;
	GLFWWindow(GLFWWindow &&) noexcept = default;

	GLFWWindow& operator=(const GLFWWindow &) = delete;
	GLFWWindow& operator=(GLFWWindow &&) noexcept = default;


};