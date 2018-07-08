#pragma once

#include "Display/Layout/Layout.h"
#include "Display/Window/Window.h"
#include "Composition/Entity/EntityManager.h"

class WindowManager
{
public:

	WindowManager(EntityManager&);
	virtual ~WindowManager() = default;

	WindowManager(const WindowManager&) = delete;
	WindowManager(WindowManager&& o) noexcept = delete;

	WindowManager& operator=(const WindowManager&) = delete;
	WindowManager& operator=(WindowManager&& other) noexcept = delete;

	/* Close operations. */
	virtual bool ShouldClose() const = 0;
	virtual void SignalClose() = 0;

	//Process. to create a window.
	virtual Window& CreateWindow(WindowParameters&) = 0;

	//Modifier operations.
	void Draw();
	void Refresh();
	void Resize(int, int);
	void WindowPos(int, int);


protected:

	//List of all windows handled by the DesktopWindowManager.
	std::list<Entity> windows_;

	EntityManager* entityManager_;

	int monitorCount_;
	bool runningFlag_;

};
