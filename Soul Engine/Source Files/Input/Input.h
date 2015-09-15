#pragma once

#include "Engine Core/BasicDependencies.h"

	bool SetKey(int, std::function<void()>);
	void InputKeyboardCallback(GLFWwindow*, int, int, int, int);
	void SetInputWindow(GLFWwindow*);
	GLFWwindow* GetInputWindow();
