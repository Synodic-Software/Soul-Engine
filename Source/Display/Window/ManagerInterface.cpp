#include "ManagerInterface.h"
#include "Utility\Logger.h"
#include "Display\Window\Implementations\Desktop\DesktopManager.h"

#define CURRENT_PLATFORM "DESKTOP"

ManagerInterface::ManagerInterface() {
	if (CURRENT_PLATFORM == "DESKTOP") {
		manager = new DesktopManager();
	} else {
		S_LOG_FATAL("The platform '", CURRENT_PLATFORM, "' is not currently supported by Soul Engine.");
	}
}

ManagerInterface::~ManagerInterface() {
	delete manager;
}

AbstractWindow* ManagerInterface::CreateWindow(WindowType type, const std::string& name, int monitor, uint x, uint y, uint width, uint height) {
	return manager->CreateWindow(type, name, monitor, x, y, width, height);
}

void ManagerInterface::SetWindowLayout(AbstractWindow* window, Layout* layout) {
	manager->SetWindowLayout(window, layout);
}


