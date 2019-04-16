#include "RasterModule.h"

#include "Display/DisplayModule.h"
#include "Modules/Vulkan/VulkanRasterBackend.h"


//TODO: This needs to instead be runtime loadable from shared libraries or statically linked
std::shared_ptr<RasterModule> RasterModule::CreateModule(
	std::shared_ptr<SchedulerModule>& scheduler,
	std::shared_ptr<DisplayModule>& displayModule)
{

	return std::make_unique<VulkanRasterBackend>(scheduler, displayModule);

}
