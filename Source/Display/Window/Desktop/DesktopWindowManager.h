#pragma once

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include "Display/Layout/Layout.h"
#include "Display/Window/SoulWindow.h"

#include "Display/Window/AbstractWindowManager.h"

class DesktopWindowManager : public AbstractWindowManager
{
public:

	DesktopWindowManager();
	~DesktopWindowManager();

	/* DesktopWindowManager can be neither copied, nor assigned. */
	DesktopWindowManager(DesktopWindowManager const&) = delete;
	void operator=(DesktopWindowManager const&) = delete;


	/* Close operations. */
	bool ShouldClose() const override;
	void SignalClose() override;

	//Process. to create a window.
	SoulWindow* CreateWindow(WindowParameters&) override;

	//Modifier operations.
	void Draw();
	void Refresh();
	void Resize(int, int);
	void WindowPos(int, int);

private:

};
