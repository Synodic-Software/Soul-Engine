#pragma once

#include <vulkan\vulkan.hpp>
#include <GLFW\glfw3.h>

#include <Display\Window\WindowManager.h>
#include "GLFW.h"

class GLFWManager : public WindowManager
{
public:
	static GLFWManager& Instance() {
		static GLFWManager instance;
		return instance;
	}

	GLFWManager(GLFWManager const&) = delete;
	void operator=(GLFWManager const&) = delete;

	bool ShouldClose();
	void SignalClose();

	GLFW* CreateWindow(WindowType, const std::string&, int moniotr, uint x, uint y, uint width, uint height);
	
	void SetWindowLayout(GLFW*, Layout*);

	void Draw();
	void Resize(GLFWwindow *, int, int);
	void Refresh(GLFWwindow*);
	void WindowPos(GLFWwindow *, int, int);

	void Close(GLFWwindow *);


private:
	GLFWManager();
	~GLFWManager();
};