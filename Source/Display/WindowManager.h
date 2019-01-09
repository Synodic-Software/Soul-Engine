#pragma once

#include "Display/Window/Window.h"
#include "Composition/Entity/Entity.h"

#include <vector>

class WindowManager
{
public:

	WindowManager();
	virtual ~WindowManager() = default;
	virtual void Terminate() = 0;

	WindowManager(const WindowManager&) = delete;
	WindowManager(WindowManager&& o) noexcept = default;

	WindowManager& operator=(const WindowManager&) = delete;
	WindowManager& operator=(WindowManager&& other) noexcept = default;

	//Close operations.
	virtual bool ShouldClose() const = 0;
	virtual void SignalClose() = 0;

	//Process to create a window.
	virtual Window& CreateWindow(WindowParameters&) = 0;

	//Modifier operations.
	virtual void Draw() = 0;
	void Refresh();
	void Resize(int, int);
	void WindowPos(int, int);


protected:

	//List of all windows handled by the DesktopWindowManager.
	std::vector<Entity> windows_;

	int monitorCount_;
	bool runningFlag_;

};
