#include "InputState.h"
#include <list>
#include "Multithreading\Scheduler.h"


std::list<std::function<void()> > keyHash[350];


bool IsKeyAvailable(int key){
	return keyHash[key].size() == 0;
}

void InputState::InputKeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
	if (!IsKeyAvailable(key)){
		for (std::list<std::function<void()> >::iterator itr = keyHash[key].begin(); itr != keyHash[key].end(); itr++){
			(*itr)();
		}
	}
}

bool InputState::SetKey(int key, std::function<void()> function){
	
	keyHash[key].push_back(function);
	return true;
}

void InputState::InputMouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	int width, height;

	//Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&]() {

		glfwGetWindowSize(window, &width, &height);

		if (ResetMouse) {
			glfwSetCursorPos(window, width / 2.0f, height / 2.0f);
		}
	//});

	//Scheduler::Block();
	xPos = xpos - (width / 2.0);
	yPos = ypos - (height / 2.0);
}

void InputState::InputScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	scrollXOffset = xoffset;
	scrollYOffset = yoffset;
}

void InputState::ResetOffsets(){
	scrollXOffset = 0.0f;
	scrollYOffset = 0.0f;
	if (ResetMouse) {
		xPos = 0;
		yPos = 0;
	}
}