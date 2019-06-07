#include "EntityRenderGraphBackend.h"

#include "Core/Utility/Exception/Exception.h"
#include "Render/Raster/RasterModule.h"


EntityRenderGraphBackend::EntityRenderGraphBackend(
	std::shared_ptr<RasterModule>& rasterModule, 
	std::shared_ptr<SchedulerModule>& scheduler):
	RenderGraphModule(rasterModule, scheduler),
	rasterModule_(rasterModule), 
	entityRegistry_(new EntityRegistry),
	builder_(rasterModule_, entityRegistry_)
{
}

void EntityRenderGraphBackend::Execute()
{

	for (const auto& callback : graphTasks_) {

		CommandList commandList;
		callback(*entityRegistry_, commandList);

		rasterModule_->Compile(commandList);
		rasterModule_->ExecutePass(commandList);

	}

}


void EntityRenderGraphBackend::CreateRenderPass(RenderTaskParameters& parameters,
	std::function<std::function<void(const EntityRegistry&, CommandList&)>(RenderGraphBuilder&)>
		passCallback)
{

	//GraphTask& task = renderGraph_.CreateTask();

	//Call the pass construction
	auto callback = passCallback(builder_);

	//Store the execution step for later
	graphTasks_.push_back(callback);

}