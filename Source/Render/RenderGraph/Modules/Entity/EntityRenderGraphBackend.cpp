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

	for (const auto& [pass, surfaces, callback] : graphTasks_) {

		CommandList commandList;
		callback(*entityRegistry_, commandList);

		rasterModule_->Compile(commandList);

		for (const auto& surface : surfaces) {
			rasterModule_->ExecutePass(pass, surface, commandList);
		}

	}

}


void EntityRenderGraphBackend::CreateRenderPass(RenderTaskParameters& parameters,
	std::function<std::function<void(const EntityRegistry&, CommandList&)>(RenderGraphBuilder&)>
		passCallback)
{

	std::function<void(const EntityRegistry&, CommandList&)> callback;

	//Create the renderpass
	Entity pass = rasterModule_->CreatePass([&](Entity pass) {
		RenderGraphBuilder builder(rasterModule_, entityRegistry_, pass, false);

		// Call the pass construction
		callback = passCallback(builder);

	});

	//Store the execution step for later
	graphTasks_.push_back({pass, {}, callback});

}