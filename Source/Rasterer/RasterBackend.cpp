#include "RasterBackend.h"

#include "Modules/Vulkan/VulkanRasterBackend.h"

std::shared_ptr<RasterBackend> RasterBackend::CreateModule()
{

	return std::make_shared<VulkanRasterBackend>();

}
