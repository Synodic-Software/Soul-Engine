#include "RasterBackend.h"

#include "Display/Display.h"
#include "Modules/Vulkan/VulkanRasterBackend.h"

std::unique_ptr<Display> RasterBackend::displayModule_ = nullptr;

//TODO: This needs to instead be runtime loadable from shared libraries or statically linked
std::unique_ptr<RasterBackend> RasterBackend::CreateModule()
{
	displayModule_ = Display::CreateModule();
	return std::make_unique<VulkanRasterBackend>(*displayModule_);

}
