#include "VulkanRenderGraphBackend.h"

#include "Core/Utility/Exception/Exception.h"

VulkanRenderGraphBackend::VulkanRenderGraphBackend(std::shared_ptr<RasterModule>& rasterModule):
	rasterModule_(rasterModule)
{
}

void VulkanRenderGraphBackend::Execute()
{

	for (const auto& callback : graphTasks_) {
		CommandList commandList(rasterModule_);
		callback(graphRegistry_, commandList);
	}

}


void VulkanRenderGraphBackend::CreatePass(std::string name,
	std::function<std::function<void(EntityReader&, CommandList&)>(EntityWriter&)> passCallback)
{

	graphTasks_.push_back(passCallback(graphRegistry_));

}