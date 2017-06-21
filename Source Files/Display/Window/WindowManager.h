#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Metrics.h"
#include <string>
#include <functional>
#include "Window.h"
#include "Display\Layout\Layout.h"

#ifdef WIN32
#undef CreateWindow
#endif

/* . */
/* . */
namespace WindowManager {

	/*
	 *    GLFW needs to be initialized.
	 *
	 *    @param [in,out]	parameter1	If non-null, true to parameter 1.
	 */

	void Initialize(bool*);

	/* cleanup all windows. */
	/* Terminates this object. */
	void Terminate();

	/*
	 *    Determine if we should close.
	 *
	 *    @return	True if it succeeds, false if it fails.
	 */

	bool ShouldClose();

	/* Signel close. */
	/* Signel close. */
	void SignelClose();

	/*
	 *    Creates a window.
	 *
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 *    @param	monitor   	The monitor.
	 *    @param	x		  	An uint to process.
	 *    @param	y		  	An uint to process.
	 *    @param	width	  	The width.
	 *    @param	height	  	The height.
	 *
	 *    @return	Null if it fails, else the new window.
	 */

	Window* CreateWindow(WindowType, const std::string&, int monitor, uint x, uint y, uint width, uint height);

	/*
	 *    Sets window layout.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param [in,out]	parameter2	If non-null, the second parameter.
	 */

	void SetWindowLayout(Window*, Layout*);

	/* Draws this object. */
	/* Draws this object. */
	void Draw();

	/*
	 *    callbacks.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 *    @param 		 	parameter3	The third parameter.
	 */

	void Resize(GLFWwindow *, int, int);

	/*
	 *    Refreshes the given parameter 1.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void Refresh(GLFWwindow*);

	/*
	 *    Window position.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 *    @param 		 	parameter3	The third parameter.
	 */

	void WindowPos(GLFWwindow *, int, int);

	/*
	 *    Closes the given parameter 1.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void Close(GLFWwindow *);
}