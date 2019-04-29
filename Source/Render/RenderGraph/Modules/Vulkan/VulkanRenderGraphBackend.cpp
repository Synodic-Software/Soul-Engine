#include "VulkanRenderGraphBackend.h"

#include "Core/Utility/Exception/Exception.h"

VulkanRenderGraphBackend::VulkanRenderGraphBackend(std::shared_ptr<RasterModule>& rasterModule)
{
}

void VulkanRenderGraphBackend::CreatePass(std::string name,
	std::function<std::function<void(EntityReader&, CommandList&)>(EntityWriter&)> passCallback)
{

	graphTasks_.push_back(passCallback(graphRegistry_));

}