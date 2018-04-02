#pragma once

#include "Display\Window\AbstractManager.h"

class DesktopManager : public AbstractManager
{
public:
	static AbstractManager& Instance() {
		static DesktopManager instance;
		return instance;
	}

	/* GLFWManager can be neither copied, nor assigned. */
	DesktopManager(DesktopManager const&) = delete;
	void operator=(DesktopManager const&) = delete;

	/* Close operations. */
	bool ShouldClose();
	void SignalClose();
	void Close(AbstractManager*);

	/* Process. to create a window. */
	AbstractWindow* CreateWindow(WindowType, const std::string&, int monitor, uint x, uint y, uint width, uint height);

	/* Set the Window's Layout. */
	void SetWindowLayout(AbstractManager*, Layout*);

	/* Modifier operations. */
	void Draw();
	void Resize(AbstractWindow*, int, int);
	void Refresh(AbstractManager*);
	void WindowPos(AbstractManager*, int, int);

private:
	/* Constructor. */
	DesktopManager();

	/* Destructor. */
	~DesktopManager();

};