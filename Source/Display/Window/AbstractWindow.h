#pragma once

#include <vulkan\vulkan.hpp>
#include <GLFW\glfw3.h>

#include "Core/Utility/Types.h"
#include <Display\Layout\Layout.h>

#include <string>
#include <memory>

class AbstractWindow
{
public:
	/* Constructor. */
	AbstractWindow(WindowType, const std::string&, uint x, uint y, uint width, uint height, void*, void*);

	/* Destructor. */
	~AbstractWindow() = default;

	/* Draws this object. */
	virtual void Draw() = 0;

	/* Handle of the window. */
	void* windowHandle;

	/* The layout */
	std::unique_ptr<Layout> layout;

	/* Type of the window */
	WindowType windowType;
	/* The title */
	std::string title;
	/* The position */
	uint xPos;
	/* The position */
	uint yPos;
	/* The width */
	uint width;
	/* The height */
	uint height;
};