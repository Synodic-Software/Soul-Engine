#include "InputState.h"
#include <list>

std::list<std::function<void(int)> > keyHash[350];
std::list<std::function<void(double, double)> > mouseEvents;

bool IsKeyAvailable(int key) {
	return keyHash[key].size() == 0;
}

void InputState::InputKeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (!IsKeyAvailable(key)) {
		for (std::list<std::function<void(int)> >::iterator itr = keyHash[key].begin(); itr != keyHash[key].end(); itr++) {
			(*itr)(action);
		}
	}
}

bool InputState::SetKey(int key, std::function<void(int)> function) {

	keyHash[key].push_back(function);
	return true;

}

bool InputState::AddMouseCallback(std::function<void(double, double)> function) {

	mouseEvents.push_back(function);
	return true;

}


//called from glfwPollEvents which is run only on main
void InputState::InputMouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	int width, height;

	glfwGetWindowSize(window, &width, &height);

	if (ResetMouse) {
		glfwSetCursorPos(window, width / 2.0f, height / 2.0f);
	}

	xPos = xpos - (width / 2.0);
	yPos = ypos - (height / 2.0);

	for (std::list<std::function<void(double, double)> >::iterator itr = mouseEvents.begin(); itr != mouseEvents.end(); itr++) {
		(*itr)(xPos, yPos);
	}
}

void InputState::InputScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	scrollXOffset = xoffset;
	scrollYOffset = yoffset;
}

void InputState::ResetOffsets() {
	scrollXOffset = 0.0f;
	scrollYOffset = 0.0f;
	if (ResetMouse) {
		xPos = 0;
		yPos = 0;
	}
}