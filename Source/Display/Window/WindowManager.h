#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Metrics.h"
#include "Window.h"
#include "Display\Layout\Layout.h"

class WindowManager {

public:

	static WindowManager& Instance()
	{
		static WindowManager instance;
		return instance;
	}

	WindowManager(WindowManager const&) = delete;
	void operator=(WindowManager const&) = delete;


	bool ShouldClose();
	void SignelClose();

	Window* CreateWindow(WindowType, const std::string&, int monitor, uint x, uint y, uint width, uint height);

	void SetWindowLayout(Window*, Layout*);

	void Draw();
	void Resize(GLFWwindow *, int, int);
	void Refresh(GLFWwindow*);
	void WindowPos(GLFWwindow *, int, int);

	void Close(GLFWwindow *);


private:

	WindowManager();
	~WindowManager();

	std::list<std::unique_ptr<Window>> windows;
	Window* masterWindow = nullptr;

	int monitorCount;
	GLFWmonitor** monitors;

	bool runningFlag;
};