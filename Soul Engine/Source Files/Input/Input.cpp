#include "Input.h"

std::list<std::function<void()> > keyHash[350];
GLFWwindow* window;

bool IsKeyAvailable(int key){
	return keyHash[key].size() == 0;
}

void InputKeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
	if (!IsKeyAvailable(key)){
		for (std::list<std::function<void()> >::iterator itr = keyHash[key].begin(); itr != keyHash[key].end(); itr++){
		(*itr)();
		}
	}
}

bool SetKey(int key, std::function<void()> function){
	//doesn't currently check for availability, this is bad - talk to me about whats needed here
	keyHash[key].push_back(function);
	return true;
}
void SetInputWindow(GLFWwindow* windowN){
	window = windowN;
}
GLFWwindow* GetInputWindow(){
	return window;
}
