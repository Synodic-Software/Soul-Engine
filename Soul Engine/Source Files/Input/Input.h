#pragma once

#include "Utility\Vulkan\VulkanBackend.h"

class Input
{


	virtual void InputKeyboardCallback(GLFWwindow*, int, int, int, int) = 0;
	virtual void InputMouseCallback(GLFWwindow*, double, double) = 0;
	virtual void InputScrollCallback(GLFWwindow*, double, double) = 0;

	static Input *inputHandle;

public:

	virtual void setEventHandling() final { inputHandle = this; }

	static void KeyCallback(//key presses?
		GLFWwindow *window,
		int key,
		int scancode,
		int action,
		int mods)
	{
		if (inputHandle)
			inputHandle->InputKeyboardCallback(window, key, scancode, action, mods);
	}

	static void MouseCallback(
		GLFWwindow* window, 
		double xoffset,
		double yoffset)
	{
		if (inputHandle)
			inputHandle->InputMouseCallback(window, xoffset, yoffset);
	}

	static void ScrollCallback(
		GLFWwindow* window, 
		double xoffset, 
		double yoffset)
	{
		if (inputHandle)
			inputHandle->InputScrollCallback(window, xoffset, yoffset);
	}
};