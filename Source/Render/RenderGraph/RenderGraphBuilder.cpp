#include "RenderGraphBuilder.h"

RenderGraphBuilder::RenderGraphBuilder(std::shared_ptr<RasterModule>& rasterModule,
	std::shared_ptr<EntityRegistry>& entityRegistry):
	entityRegistry_(entityRegistry),
	rasterModule_(rasterModule)
{
}

void RenderGraphBuilder::CreateOutput(RenderGraphOutputParameters&)
{
}

void RenderGraphBuilder::CreateInput(RenderGraphInputParameters&)
{
}

void RenderGraphBuilder::CreateSubpass()
{

}

Entity RenderGraphBuilder::View()
{

	Entity returnedEntity = entityRegistry_->CreateEntity();
	entityRegistry_->AttachComponent<RenderView>(returnedEntity);

	return returnedEntity;
}
