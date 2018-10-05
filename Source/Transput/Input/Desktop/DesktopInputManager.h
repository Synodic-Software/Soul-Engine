#pragma once

#include "Composition/Event/EventManager.h"
#include "Transput/Input/Key.h"
#include "Transput/Input/InputManager.h"

#include "GLFW/glfw3.h"

class DesktopInputManager : public InputManager {

public:

	DesktopInputManager(EventManager&);
	~DesktopInputManager() override = default;

	DesktopInputManager(const DesktopInputManager&) = delete;
	DesktopInputManager(DesktopInputManager&& o) = default;

	DesktopInputManager& operator=(const DesktopInputManager&) = delete;
	DesktopInputManager& operator=(DesktopInputManager&& other) noexcept = default;

	//void AttachWindow(DesktopWindow* window);
	bool Poll() override;


private:

	void KeyCallback(GLFWwindow*, int, int, int, int);
	void CharacterCallback(GLFWwindow*, uint);
	void ModdedCharacterCallback(GLFWwindow*, uint, int);
	void ButtonCallback(GLFWwindow*, int, int, int);
	void CursorCallback(GLFWwindow*, double, double);
	void CursorEnterCallback(GLFWwindow*, int);
	void ScrollCallback(GLFWwindow*, double, double);

	std::unordered_map<int, Key> keyStates_;

	double mouseXPos_;
	double mouseYPos_;

	double mouseXOffset_;
	double mouseYOffset_;

	bool firstMouse_;

};
