#pragma once

#include "Utility\Vulkan\VulkanBackend.h"
#include <functional>
#include "Input.h"

class InputState : Input{

public:
	static InputState& GetInstance()
	{
		static InputState instance;
		instance.setEventHandling();
		return instance;
	}

	bool SetKey(int, std::function<void()>);
	void ResetOffsets();

	bool ResetMouse;
	double xPos;
	double yPos;
	double scrollXOffset;
	double scrollYOffset;

private:
	InputState() {}

//protected:
	void InputKeyboardCallback(GLFWwindow*, int, int, int, int);
	void InputMouseCallback(GLFWwindow*, double, double);
	void InputScrollCallback(GLFWwindow*, double, double);


public:
	InputState(InputState const&) = delete;
	void operator=(InputState const&) = delete;
};
