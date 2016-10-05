#pragma once

#include "Engine Core/BasicDependencies.h"
#include <GLFW/glfw3.h>

namespace SoulInput{
	bool SetKey(int, std::function<void()>);
	void InputKeyboardCallback(GLFWwindow*, int, int, int, int);
	void SetInputWindow(GLFWwindow*);
	GLFWwindow* GetInputWindow();
	void UpdateMouseCallback(GLFWwindow*, double, double);

	bool ResetMouse = false;
	double xPos=0.0;
	double yPos=0.0;
}
