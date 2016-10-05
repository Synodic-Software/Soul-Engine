#include "Input.h"



std::list<std::function<void()> > keyHash[350];
GLFWwindow* window;

bool IsKeyAvailable(int key){
	return keyHash[key].size() == 0;
}

void SoulInput::InputKeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
	if (!IsKeyAvailable(key)){
		for (std::list<std::function<void()> >::iterator itr = keyHash[key].begin(); itr != keyHash[key].end(); itr++){
		(*itr)();
		}
	}
}

bool SoulInput::SetKey(int key, std::function<void()> function){
	//doesn't currently check for availability, this is bad - talk to me about whats needed here
	keyHash[key].push_back(function);
	return true;
}
void SoulInput::SetInputWindow(GLFWwindow* windowN){
	window = windowN;
}
GLFWwindow* SoulInput::GetInputWindow(){
	return window;
}

void SoulInput::UpdateMouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	xPos = xpos - (width / 2.0);
	yPos = ypos - (height / 2.0);

	if (ResetMouse){
		glfwSetCursorPos(window, width / 2.0f, height / 2.0f);
	}
}