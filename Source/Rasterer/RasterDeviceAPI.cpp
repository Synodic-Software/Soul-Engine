#include "RasterDeviceAPI.h"

#include "Display/Window/GLFW/GLFWModule.h"
#include "Display/Window/Mock/MockModule.h"

std::shared_ptr<RasterDeviceAPI> RasterDeviceAPI::CreateModule()
{
	return nullptr;
}