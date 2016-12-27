#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Metrics.h"

#include <string>

typedef enum WindowType { WINDOWED, FULLSCREEN, BORDERLESS };

class Window
{
public:
	Window(const std::string&, uint x, uint y, uint width, uint height, GLFWmonitor*);
	~Window();

	GLFWwindow* windowHandle;
	
	void Draw();

	const std::string& title;
	int xPos; 
	int yPos; 
	int width;
	int height;
};

