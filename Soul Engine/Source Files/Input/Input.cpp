#include "Input.h"

std::list<std::function<void()> > keyHash[350];
GLFWwindow* window;

bool IsKeyAvailable(int key){
	if (keyHash[key].size() == 0){
		return true;
	}
	else{
		return false;
	}
}
void InputKeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
	if (!IsKeyAvailable(key)){
		for (std::list<std::function<void()> >::iterator itr = keyHash[key].begin(); itr != keyHash[key].end(); itr++){
		(*itr)();
		}

	}
}
bool SetKey(int key, std::function<void()> function){
	//if (IsKeyAvailable(key)){
		keyHash[key].push_back(function);
		return true;
	//}
	//else {
	//	return false;
	//}
}
void SetInputWindow(GLFWwindow* windowN){
	window = windowN;
}
GLFWwindow* GetInputWindow(){
	return window;
}
