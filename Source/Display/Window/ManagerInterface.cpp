
#include "Utility\Logger.h"
#include "ManagerInterface.h"
#include "Display\Window\Implementations\Desktop\DesktopManager.h"

/* [ PLACE HOLDER SOLUTION TO PLATFORM SELECTION ] */
#define CURRENT_PLATFORM "DESKTOP"

/* 
	This is an interface. 
	The majority of the functions in this file refer to the methods of the files with which they interface.
	Those methods are defined in their respective classes.
	For more details on desktop-specific processes, see the following files:
		- Display\Window\Implementations\Desktop\DesktopManager.h
		- Display\Window\Implementations\Desktop\DesktopWindow.h
	There no current support for other platfroms.

*/

/* 
CONSTRUCTOR. 
*/
ManagerInterface::ManagerInterface() {
	if (CURRENT_PLATFORM == "DESKTOP") {
		manager = &DesktopManager::Instance();
	} else {
		S_LOG_FATAL("The platform '", CURRENT_PLATFORM, "' is not currently supported by Soul Engine.");
	}
}


/*
DESTRUCTOR.
*/
ManagerInterface::~ManagerInterface() {
	delete manager;
}


/* 
PROCESS TO CREATE A WINDOW.
*/
AbstractWindow* ManagerInterface::CreateWindow(WindowType type, const std::string& name, int monitor, uint x, uint y, uint width, uint height) {
	return DesktopManager::Instance().CreateWindow(type, name, monitor, x, y, width, height);
}


/* 
SET THE WINDOW'S LAYOUT.
*/
void ManagerInterface::SetWindowLayout(AbstractWindow* window, Layout* layout) {
	DesktopManager::Instance().SetWindowLayout(window, layout);
}


/* 

CLOSE OPERATIONS. 

*/
bool ManagerInterface::ShouldClose() {
	return DesktopManager::Instance().ShouldClose();
}

void ManagerInterface::SignalClose() {
	DesktopManager::Instance().SignalClose();
}

void ManagerInterface::Close(void* handler) {
	DesktopManager::Instance().Close(handler);
}

/*

MODIFIER OPERATIONS.

*/

void ManagerInterface::Draw() {
	DesktopManager::Instance().Draw();
}

void ManagerInterface::Refresh(void* handler) {
	DesktopManager::Instance().Refresh(handler);
}

void ManagerInterface::Resize(void* handler, int width, int height) {
	DesktopManager::Instance().Resize(handler, width, height);
}


void ManagerInterface::WindowPos(void* handler, int x, int y) {
	DesktopManager::Instance().WindowPos(handler, x, y);
}



