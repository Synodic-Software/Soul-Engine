#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Metrics.h"

typedef enum WindowType { WINDOWED, FULLSCREEN, BORDERLESS };

class Window
{
public:
	Window( uint x, uint y, uint width, uint height, GLFWmonitor*);
	~Window();

	GLFWwindow* windowHandle;
	
	void Draw();
};

