#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Metrics.h"
#include "Display\Layout\Layout.h"

#include <string>
#include <memory>

/* A window. */
/* A window. */
class Window
{
public:

	/*
	 *    Constructor.
	 *
	 *    @param 		 	parameter1	The first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 *    @param 		 	x		  	An uint to process.
	 *    @param 		 	y		  	An uint to process.
	 *    @param 		 	width	  	The width.
	 *    @param 		 	height	  	The height.
	 *    @param [in,out]	parameter7	If non-null, the parameter 7.
	 *    @param [in,out]	parameter8	If non-null, the parameter 8.
	 */

	Window(WindowType,const std::string&, uint x, uint y, uint width, uint height, GLFWmonitor*, GLFWwindow*);
	/* Destructor. */
	/* Destructor. */
	~Window();

	/* Handle of the window */
	/* Handle of the window */
	GLFWwindow* windowHandle;

	/* Draws this Window. */
	/* Draws this Window. */
	void Draw();

	/* The layout */
	/* The layout */
	std::unique_ptr<Layout> layout;

	/* Type of the window */
	/* Type of the window */
	WindowType windowType;
	/* The title */
	/* The title */
	std::string title;
	/* The position */
	/* The position */
	uint xPos;
	/* The position */
	/* The position */
	uint yPos;
	/* The width */
	/* The width */
	uint width;
	/* The height */
	/* The height */
	uint height;
};

