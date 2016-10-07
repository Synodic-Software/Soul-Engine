#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility\Vulkan\VulkanBackend.h"

namespace SoulInput{
	bool SetKey(int, std::function<void()>);
	void InputKeyboardCallback(GLFWwindow*, int, int, int, int);
	void UpdateMouseCallback(GLFWwindow*, double, double);
	void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	void ResetOffsets();

	bool ResetMouse = false;
	double xPos=0.0;
	double yPos=0.0;
	double scrollXOffset = 0.0f;
	double scrollYOffset = 0.0f;
}
