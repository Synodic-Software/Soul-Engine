#pragma once

#include "Display/Window/WindowManager.h"
#include "Display/Window/Window.h"
#include "Transput/Input/Desktop/DesktopInputManager.h"

class DesktopWindowManager : public WindowManager
{
public:

	DesktopWindowManager(EntityManager&, DesktopInputManager&);
	~DesktopWindowManager() override;

	DesktopWindowManager(const DesktopWindowManager&) = delete;
	DesktopWindowManager(DesktopWindowManager&& o) noexcept = delete;

	DesktopWindowManager& operator=(const DesktopWindowManager&) = delete;
	DesktopWindowManager& operator=(DesktopWindowManager&& other) noexcept = delete;


	/* Close operations. */
	bool ShouldClose() const override;
	void SignalClose() override;

	//Process. to create a window.
	Window& CreateWindow(WindowParameters&) override;

	//Modifier operations.
	void Draw();
	void Refresh();
	void Resize(int, int);
	void WindowPos(int, int);

private:

	Entity masterWindow_;

	DesktopInputManager * inputManager_;
	GLFWmonitor** monitors_;

};
