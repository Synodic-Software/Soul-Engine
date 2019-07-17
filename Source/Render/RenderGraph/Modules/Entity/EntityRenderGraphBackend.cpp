#include "EntityRenderGraphBackend.h"

#include "Core/Utility/Exception/Exception.h"
#include "Render/Raster/RasterModule.h"


EntityRenderGraphBackend::EntityRenderGraphBackend(
	std::shared_ptr<RasterModule>& rasterModule, 
	std::shared_ptr<SchedulerModule>& scheduler):
	RenderGraphModule(rasterModule, scheduler),
	rasterModule_(rasterModule), 
	entityRegistry_(new EntityRegistry)
{
}

void EntityRenderGraphBackend::Execute()
{

	for (const auto& [pass, callback] : graphTasks_) {

		CommandList commandList;
		callback(*entityRegistry_, commandList);

		rasterModule_->Compile(commandList);
		rasterModule_->ExecutePass(pass, commandList);

	}

}


void EntityRenderGraphBackend::CreateRenderPass(RenderTaskParameters& parameters,
	std::function<std::function<void(const EntityRegistry&, CommandList&)>(RenderGraphBuilder&)>
		passCallback)
{

	//GraphTask& task = renderGraph_.CreateTask();

	//Create the renderpass;
	Entity pass = rasterModule_->RegisterPass();
	RenderGraphBuilder builder(rasterModule_, entityRegistry_, pass, false);

	//Call the pass construction
	auto callback = passCallback(builder);

	//Store the execution step for later
	graphTasks_.push_back({pass, callback});

}