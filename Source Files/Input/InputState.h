#pragma once

#include "Utility\Includes\GLFWIncludes.h"
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

	bool SetKey(int, std::function<void(int)>);
	bool AddMouseCallback(std::function<void(double,double)>);
	void ResetOffsets();

	bool ResetMouse; //boolean for if mouse should be reset
	double xPos; //x position of mouse
	double yPos; //y position of mouse
	double scrollXOffset; //x offset of scroll
	double scrollYOffset; //y offset of scroll

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
