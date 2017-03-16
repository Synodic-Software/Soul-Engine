#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Metrics.h"
#include "Display\Layout\Layout.h"

#include <string>
#include <memory>

class Window
{
public:
	Window(WindowType,const std::string&, uint x, uint y, uint width, uint height, GLFWmonitor*, GLFWwindow*);
	~Window();

	GLFWwindow* windowHandle;

	void Draw();

	std::unique_ptr<Layout> layout;

	WindowType windowType;
	std::string title;
	uint xPos;
	uint yPos;
	uint width;
	uint height;
};

