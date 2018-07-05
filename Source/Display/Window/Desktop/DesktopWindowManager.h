#pragma once

#include "Display/Window/WindowManager.h"
#include "Display/Window/Window.h"
#include "Transput/Input/Desktop/DesktopInputManager.h"

class DesktopWindowManager : public WindowManager
{
public:

	DesktopWindowManager(DesktopInputManager&);
	~DesktopWindowManager() override;

	DesktopWindowManager(DesktopWindowManager const&) = delete;	
	DesktopWindowManager(DesktopWindowManager&& o) = delete;

	DesktopWindowManager& operator=(DesktopWindowManager const&) = delete;
	DesktopWindowManager& operator=(DesktopWindowManager&& other) = delete;
	

	/* Close operations. */
	bool ShouldClose() const override;
	void SignalClose() override; 

	//Process. to create a window.
	Window* CreateWindow(WindowParameters&) override;

	//Modifier operations.
	void Draw();
	void Refresh();
	void Resize(int, int);
	void WindowPos(int, int);

private:

	DesktopInputManager* inputManager_;
	GLFWmonitor** monitors_;

};
