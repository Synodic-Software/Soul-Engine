#pragma once
#include "Core/Utility/Types.h"

#include "Display\Window\AbstractManager.h"
#include <memory>

/*
This is an interface.
The majority of the functions in this file refer to the methods of the files with which they interface.
Those methods are defined in their respective classes.
For more details on desktop-specific processes, see the following files:
- Display\Window\Implementations\Desktop\DesktopManager.h
- Display\Window\Implementations\Desktop\DesktopWindow.h
There no current support for other platfroms.

*/

class AbstractManager;

class ManagerInterface
{
public:
	static ManagerInterface& Instance() {
		static ManagerInterface instance;
		return instance;
	}

	/* Process to create a window. */
	AbstractWindow* CreateWindow(WindowType, const std::string&, int monitor, uint x, uint y, uint width, uint height);

	/* Set the Window's Layout. */
	void SetWindowLayout(AbstractWindow*, Layout*);

	/* Close operations. */
	bool ShouldClose();
	void SignalClose();
	void Close(void*);


	/* Modifier operations. */
	void Draw();
	void Resize(void*, int, int);
	void Refresh(void*);
	void WindowPos(void*, int, int);


private:
	/* Pointer to the manager. */
	std::unique_ptr<AbstractManager> manager;

	/* Constructor. */
	ManagerInterface();

	/* Destructor. */
	~ManagerInterface() = default;

};