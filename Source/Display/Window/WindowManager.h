#pragma once

#include "Display/Layout/Layout.h"
#include "Display/Window/Window.h"

class WindowManager
{
public:

	WindowManager();
	virtual ~WindowManager() = default;

	WindowManager(WindowManager const&) = delete;
	void operator=(WindowManager const&) = delete;

	WindowManager(WindowManager&& o) = delete;
	WindowManager& operator=(WindowManager&& other) = delete;

	/* Close operations. */
	virtual bool ShouldClose() const = 0;
	virtual void SignalClose() = 0;

	//Process. to create a window.
	virtual Window* CreateWindow(WindowParameters&) = 0;

	//Modifier operations.
	void Draw();
	void Refresh();
	void Resize(int, int);
	void WindowPos(int, int);


protected:

	//List of all windows handled by the DesktopWindowManager.
	std::list<std::unique_ptr<Window>> windows_;

	Window* masterWindow_;

	int monitorCount_;
	bool runningFlag_;

};