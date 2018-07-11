#pragma once

#include "Display/Window/WindowManager.h"
#include "Display/Window/Window.h"
#include "Transput/Input/Desktop/DesktopInputManager.h"

class DesktopWindowManager : public WindowManager
{

public:

	DesktopWindowManager(DesktopInputManager&);
	~DesktopWindowManager() override = default;
	void Terminate() override;

	DesktopWindowManager(const DesktopWindowManager&) = delete;
	DesktopWindowManager(DesktopWindowManager&& o) noexcept = default;

	DesktopWindowManager& operator=(const DesktopWindowManager&) = delete;
	DesktopWindowManager& operator=(DesktopWindowManager&& other) noexcept = default;


	// Close operations. 
	bool ShouldClose() const override;
	void SignalClose() override;

	//Process. to create a window.
	Window& CreateWindow(WindowParameters&) override;


private:

	Window* masterWindow_;

	DesktopInputManager * inputManager_;
	GLFWmonitor** monitors_;

};
