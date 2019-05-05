#include "EntityRenderGraphBackend.h"

#include "Core/Utility/Exception/Exception.h"
#include "Render/Raster/RasterModule.h"


EntityRenderGraphBackend::EntityRenderGraphBackend(
	std::shared_ptr<RasterModule>& rasterModule, 
	std::shared_ptr<SchedulerModule>& scheduler):
	RenderGraphModule(rasterModule, scheduler),
	rasterModule_(rasterModule),
	renderGraph_(scheduler)
{
}

void EntityRenderGraphBackend::Execute()
{

	for (const auto& callback : graphTasks_) {
		CommandList commandList(rasterModule_);
		callback(registry_, commandList);
		rasterModule_->Consume(commandList);
	}

}


void EntityRenderGraphBackend::CreatePass(std::string name,
	std::function<std::function<void(const EntityRegistry&, CommandList&)>(Graph&)> passCallback)
{

	graphTasks_.push_back(passCallback(renderGraph_));

}