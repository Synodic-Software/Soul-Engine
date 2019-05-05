#include "EntityRenderGraphBackend.h"

#include "Core/Utility/Exception/Exception.h"

EntityRenderGraphBackend::EntityRenderGraphBackend(
	std::shared_ptr<RasterModule>& rasterModule, 
	std::shared_ptr<SchedulerModule>& scheduler):
	RenderGraphModule(rasterModule, scheduler),
	renderGraph_(scheduler)
{
}

void EntityRenderGraphBackend::Execute()
{

	for (const auto& callback : graphTasks_) {
		CommandList commandList;
		callback(renderGraph_, commandList);
	}

}


void EntityRenderGraphBackend::CreatePass(std::string name,
	std::function<std::function<void(const Graph&, CommandList&)>(Graph&)> passCallback)
{

	graphTasks_.push_back(passCallback(renderGraph_));

}