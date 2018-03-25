#pragma once

<<<<<<< HEAD
class GLFW {
public:
	GLFW();
	~GLFW();
private:
	int a;
=======
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include "Display\Window\Window.h"

class GLFW : public Window
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
	
	GLFW(WindowType, const std::string&, uint x, uint y, uint width, uint height, GLFWmonitor*, GLFWwindow*);
	
	/* Destructor. */
	~GLFW() override;

	/* Draws this object. */
	void Draw();

private:
	/* Type of the window */
	WindowType windowType;
	
>>>>>>> dev-falkom
};