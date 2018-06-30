#pragma once

#include "Display/Layout/Layout.h"
#include "Display/Window/SoulWindow.h"

class AbstractWindowManager
{
public:

	AbstractWindowManager();
	virtual ~AbstractWindowManager();

	/* DesktopWindowManager can be neither copied, nor assigned. */
	AbstractWindowManager(AbstractWindowManager const&) = delete;
	void operator=(AbstractWindowManager const&) = delete;


	/* Close operations. */
	virtual bool ShouldClose() const = 0;
	virtual void SignalClose() = 0;

	//Process. to create a window.
	virtual SoulWindow* CreateWindow(WindowParameters&) = 0;

	//Modifier operations.
	void Draw();
	void Refresh();
	void Resize(int, int);
	void WindowPos(int, int);


protected:

	//List of all windows handled by the DesktopWindowManager.
	std::list<std::unique_ptr<SoulWindow>> windows;

	SoulWindow* masterWindow;
	std::any monitors;

	int monitorCount;
	bool runningFlag;


private:

};


class WindowManagerFactory
{

public:

	WindowManagerFactory() = default;
	~WindowManagerFactory() = default;

	std::unique_ptr<AbstractWindowManager> CreateWindowManager() const;

};