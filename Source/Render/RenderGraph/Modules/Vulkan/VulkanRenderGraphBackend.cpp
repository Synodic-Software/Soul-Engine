#include "VulkanRenderGraphBackend.h"

#include "Core/Utility/Exception/Exception.h"

VulkanRenderGraphBackend::VulkanRenderGraphBackend(std::shared_ptr<RasterModule>& rasterModule)
{
}

void VulkanRenderGraphBackend::CreatePass(std::string,
	std::function<std::function<void(CommandList&)>()>)
{
}