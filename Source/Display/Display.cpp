#include "Display/Display.h"
#include "System/Platform.h"

#include "Modules/GLFW/GLFWDisplay.h"

#include <memory>

Display::Display():
	active_(true)
{	
}

std::shared_ptr<Display> Display::CreateModule() {

	if constexpr (Platform::IsDesktop()) {
		return std::make_shared<GLFWDisplay>();
	}
	else {
		return nullptr;
	}

}