#include "EntityRenderGraphBackend.h"

#include "Core/Utility/Exception/Exception.h"

EntityRenderGraphBackend::EntityRenderGraphBackend(std::shared_ptr<RasterModule>& rasterModule):
	rasterModule_(rasterModule)
{
}

void EntityRenderGraphBackend::Execute()
{

	for (const auto& callback : graphTasks_) {
		CommandList commandList;
		callback(graphRegistry_, commandList);
	}

}


void EntityRenderGraphBackend::CreatePass(std::string name,
	std::function<std::function<void(EntityReader&, CommandList&)>(EntityWriter&)> passCallback)
{

	graphTasks_.push_back(passCallback(graphRegistry_));

}