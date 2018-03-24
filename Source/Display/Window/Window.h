#pragma once

// #include <vulkan/vulkan.hpp>
// #include <GLFW/glfw3.h>

#include "Metrics.h"
#include "Display\Layout\Layout.h"

#include <string>
#include <memory>

/* A window. */
class Window
{
public:

	/*
	 *    Constructor.
	 *    @param 		 	parameter1	The first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 *    @param 		 	x		  	An uint to process.
	 *    @param 		 	y		  	An uint to process.
	 *    @param 		 	width	  	The width.
	 *    @param 		 	height	  	The height.
	 *    @param [in,out]	parameter7	If non-null, the parameter 7.
	 *    @param [in,out]	parameter8	If non-null, the parameter 8.
	 */

	Window(const std::string&, uint x, uint y, uint width, uint height);
	/* Destructor. */
	virtual ~Window() {}

	/* Draws this object. */
	virtual void Draw() = 0;

	/* The layout */
	std::unique_ptr<Layout> layout;

	/* Handle of the window */
	GLFWwindow* windowHandle;

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

