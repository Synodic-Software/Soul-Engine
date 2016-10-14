#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility\Vulkan\VulkanBackend.h"

namespace SoulInput{
	bool SetKey(int, std::function<void()>);
	void InputKeyboardCallback(GLFWwindow*, int, int, int, int);
	void UpdateMouseCallback(GLFWwindow*, double, double);
	void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	void ResetOffsets();

	extern bool ResetMouse;
	extern double xPos;
	extern double yPos;
	extern double scrollXOffset;
	extern double scrollYOffset;
}
