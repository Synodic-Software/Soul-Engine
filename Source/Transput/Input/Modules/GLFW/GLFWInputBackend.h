#pragma once

#include "Transput/Input/Button.h"
#include "Transput/Input/InputModule.h"


struct GLFWwindow;

class GLFWInputBackend : public InputModule {

public:

	GLFWInputBackend();
	~GLFWInputBackend() override = default;

	GLFWInputBackend(const GLFWInputBackend&) = delete;
	GLFWInputBackend(GLFWInputBackend&&) = default;

	GLFWInputBackend& operator=(const GLFWInputBackend&) = delete;
	GLFWInputBackend& operator=(GLFWInputBackend&&) noexcept = default;


	bool Poll() override;


private:

	void KeyCallback(GLFWwindow*, int, int, int, int);
	void CharacterCallback(GLFWwindow*, uint);
	void ModdedCharacterCallback(GLFWwindow*, uint, int);
	void ButtonCallback(GLFWwindow*, int, int, int);
	void CursorCallback(GLFWwindow*, double, double);
	void CursorEnterCallback(GLFWwindow*, int);
	void ScrollCallback(GLFWwindow*, double, double);

	std::unordered_map<int, Button> keyStates_;

	double mouseXPos_;
	double mouseYPos_;

	double mouseXOffset_;
	double mouseYOffset_;

	bool firstMouse_;

};
