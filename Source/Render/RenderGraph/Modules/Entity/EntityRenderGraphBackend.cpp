#include "EntityRenderGraphBackend.h"

#include "Core/Utility/Exception/Exception.h"
#include "Render/Raster/RasterModule.h"


EntityRenderGraphBackend::EntityRenderGraphBackend(
	std::shared_ptr<RasterModule>& rasterModule, 
	std::shared_ptr<SchedulerModule>& scheduler):
	RenderGraphModule(rasterModule, scheduler),
	rasterModule_(rasterModule)
{
}

void EntityRenderGraphBackend::Execute()
{

	for (const auto& callback : graphTasks_) {

		rasterModule_->RenderPass([&]() {
			CommandList commandList(rasterModule_);
			callback(registry_, commandList);
		});

	}

}


void EntityRenderGraphBackend::CreateTask(RenderTaskParameters& parameters,
	std::function<std::function<void(const EntityRegistry&, CommandList&)>(GraphTask&)>
		passCallback)
{

	GraphTask& task = renderGraph_.CreateTask([]() {

	});

	//task.

	//Call the pass construction
	auto callback = passCallback(task);

	//Store the execution step for later
	graphTasks_.push_back(callback);

}