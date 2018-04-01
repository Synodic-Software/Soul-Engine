#pragma once

#include "Display\Window\AbstractWindow.h"

class GLFWWindow : public AbstractWindow 
{
	/* Constructor. */
	GLFWWindow(WindowType, const std::string&, uint x, uint y, uint width, uint height, void*, void*);

	/* Destructor. */
	~GLFWWindow() = default;

	/* Draws this object. */
	void Draw();

};