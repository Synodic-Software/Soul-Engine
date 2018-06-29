#pragma once

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include "SoulWindow.h"

#include "Display/Layout/Layout.h"
#include "Display/Window/DisplayManager.h"

class DisplayManager
{
public:

	DisplayManager();
	~DisplayManager();

	/* DisplayManager can be neither copied, nor assigned. */
	DisplayManager(DisplayManager const&) = delete;
	void operator=(DisplayManager const&) = delete;


	/* Close operations. */
	bool ShouldClose() const;
	void SignalClose();

	//Process. to create a window.
	SoulWindow* CreateWindow(WindowParameters&);

	//Modifier operations.
	void Draw();
	void Refresh();
	void Resize(int, int);
	void WindowPos(int, int);


private:

	//List of all windows handled by the DisplayManager.
	std::list<std::unique_ptr<SoulWindow>> windows;

	SoulWindow* masterWindow = nullptr;

	int monitorCount;
	GLFWmonitor** monitors;

	bool runningFlag;

};