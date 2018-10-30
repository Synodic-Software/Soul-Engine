#pragma once

#include "Composition/Event/EventManager.h"
#include "Transput/Input/Key.h"
#include "Transput/Input/InputManager.h"
#include "Transput/Input/Console/CLI/CLIConsoleManager.h"

#include <variant>

#include "GLFW/glfw3.h"

class DesktopInputManager : public InputManager {

public:

	using consoleManagerVariantType = std::variant<std::monostate, CLIConsoleManager>;

	DesktopInputManager(EventManager&, Soul&);
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

	consoleManagerVariantType ConstructConsoleManager(Soul&);
	ConsoleManager* ConstructConsolePtr();

	consoleManagerVariantType consoleManagerVariant_;
	ConsoleManager* consoleManager_;

	std::unordered_map<int, Key> keyStates_;

	double mouseXPos_;
	double mouseYPos_;

	double mouseXOffset_;
	double mouseYOffset_;

	bool firstMouse_;

};
