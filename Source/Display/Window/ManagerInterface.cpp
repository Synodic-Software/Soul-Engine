
#include "Core/Utility/Logger.h"
#include "ManagerInterface.h"
#include "Display\Window\Implementations\Desktop\DesktopManager.h"

///* [ PLACE HOLDER SOLUTION TO PLATFORM SELECTION ] */
//#define CURRENT_PLATFORM "DESKTOP"

const std::string CURRENT_PLATFORM = "DESKTOP";

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
		manager.reset(new DesktopManager());
	} else {
		S_LOG_FATAL("The platform '", CURRENT_PLATFORM, "' is not currently supported by Soul Engine.");
	}
}


///*
//DESTRUCTOR.
//*/
//ManagerInterface::~ManagerInterface() {
//}


/* 
PROCESS TO CREATE A WINDOW.
*/
AbstractWindow* ManagerInterface::CreateWindow(WindowType type, const std::string& name, int monitor, uint x, uint y, uint width, uint height) {
	return manager->CreateWindow(type, name, monitor, x, y, width, height);
}


/* 
SET THE WINDOW'S LAYOUT.
*/
void ManagerInterface::SetWindowLayout(AbstractWindow* window, Layout* layout) {
	manager->SetWindowLayout(window, layout);
}


/* 

CLOSE OPERATIONS. 

*/
bool ManagerInterface::ShouldClose() {
	return manager->ShouldClose();
}

void ManagerInterface::SignalClose() {
	manager->SignalClose();
}

void ManagerInterface::Close(void* handler) {
	manager->Close(handler);
}

/*

MODIFIER OPERATIONS.

*/

void ManagerInterface::Draw() {
	manager->Draw();
}

void ManagerInterface::Refresh(void* handler) {
	manager->Refresh(handler);
}

void ManagerInterface::Resize(void* handler, int width, int height) {
	manager->Resize(handler, width, height);
}


void ManagerInterface::WindowPos(void* handler, int x, int y) {
	manager->WindowPos(handler, x, y);
}



