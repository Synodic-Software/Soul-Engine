#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Metrics.h"

#include <string>

class Window
{
public:
	Window(WindowType,const std::string&, uint x, uint y, uint width, uint height, GLFWmonitor*, GLFWwindow*);
	~Window();

	GLFWwindow* windowHandle;

	void Draw();

	WindowType windowType;
	std::string title;
	uint xPos;
	uint yPos;
	uint width;
	uint height;
};

