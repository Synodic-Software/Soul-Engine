#include "RasterBackendAPI.h"

#include "Rasterer/Backend/Mock/MockModule.h"
#include "Rasterer/Backend/Vulkan/VulkanModule.h"

std::shared_ptr<RasterBackendAPI> RasterBackendAPI::CreateModule()
{
	return nullptr;
}