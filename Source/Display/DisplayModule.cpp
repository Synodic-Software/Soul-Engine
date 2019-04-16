#include "Display/DisplayModule.h"
#include "System/Platform.h"

#include "Modules/GLFW/GLFWDisplay.h"
#include "Rasterer/Modules/Vulkan/VulkanRasterBackend.h"
#include "GUI/GUIModule.h"


DisplayModule::DisplayModule(): 
	active_(true), 
	gui_(GUIModule::CreateModule())
{	
}

//TODO: There will only ever be one DisplayModule system per Soul application. This will need to be moved to the build system per platform
std::unique_ptr<DisplayModule> DisplayModule::CreateModule() {

	if constexpr (Platform::IsDesktop()) {
		return std::make_unique<GLFWDisplay>();
	}
	else {
		return nullptr;
	}

}