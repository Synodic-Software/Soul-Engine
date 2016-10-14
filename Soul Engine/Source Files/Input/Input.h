#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility\Vulkan\VulkanBackend.h"

class SoulInput{

public:
	static SoulInput& GetInstance()
	{
		static SoulInput instance; 
		return instance;
	}

	bool SetKey(int, std::function<void()>);
	void InputKeyboardCallback(GLFWwindow*, int, int, int, int);
	void UpdateMouseCallback(GLFWwindow*, double, double);
	void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	void ResetOffsets();

	bool ResetMouse;
	double xPos;
	double yPos;
	double scrollXOffset;
	double scrollYOffset;

private:
	SoulInput() {}

public:
	SoulInput(SoulInput const&) = delete;
	void operator=(SoulInput const&) = delete;
};
