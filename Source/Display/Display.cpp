#include "Display/Display.h"
#include "System/Platform.h"

#include "Modules/GLFW/GLFWDisplay.h"
#include "Rasterer/Modules/Vulkan/VulkanRasterBackend.h"

#include <memory>

Display::Display():
	active_(true)
{	
}

//TODO: There will only ever be one display system per Soul application. This will need to be moved to the build system per platform
std::shared_ptr<Display> Display::CreateModule() {

	if constexpr (Platform::IsDesktop()) {
		return std::make_shared<GLFWDisplay>();
	}
	else {
		return nullptr;
	}

}