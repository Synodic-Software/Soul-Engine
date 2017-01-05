#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Metrics.h"

#include <string>

enum WindowType { WINDOWED, FULLSCREEN, BORDERLESS };

class Window
{
public:
	Window(const std::string&, uint x, uint y, uint width, uint height, GLFWmonitor*, GLFWwindow*);
	~Window();

	GLFWwindow* windowHandle;

	void Draw();
	void Resize(GLFWwindow *, int, int);

	std::string title;
	uint xPos;
	uint yPos;
	uint width;
	uint height;
};

