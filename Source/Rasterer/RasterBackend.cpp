#include "RasterBackend.h"

#include "Display/Display.h"
#include "Modules/Vulkan/VulkanRasterBackend.h"


//TODO: This needs to instead be runtime loadable from shared libraries or statically linked
std::shared_ptr<RasterBackend> RasterBackend::CreateModule(std::shared_ptr<FiberScheduler>& scheduler, std::shared_ptr<Display>& displayModule)
{

	return std::make_unique<VulkanRasterBackend>(scheduler, displayModule);

}
