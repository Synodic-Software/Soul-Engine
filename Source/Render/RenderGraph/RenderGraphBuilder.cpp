#include "RenderGraphBuilder.h"

RenderGraphBuilder::RenderGraphBuilder(std::shared_ptr<RasterModule>& rasterModule,
	std::shared_ptr<EntityRegistry>& entityRegistry,
	Entity renderPass,
	bool subPass):
	entityRegistry_(entityRegistry),
	rasterModule_(rasterModule), 
	renderPass_(renderPass), 
	subPass_(subPass)
{
}

void RenderGraphBuilder::CreateOutput(RenderGraphOutputParameters& parameters)
{

	rasterModule_->CreatePassOutput(renderPass_, parameters.resource, Format::RGBA);

}

void RenderGraphBuilder::CreateInput(RenderGraphInputParameters& parameters)
{

	rasterModule_->CreatePassInput(renderPass_, parameters.resource, Format::RGBA);

}

void RenderGraphBuilder::CreateSubpass()
{

	throw NotImplemented();

}

Entity RenderGraphBuilder::View()
{

	Entity returnedEntity = entityRegistry_->CreateEntity();
	entityRegistry_->AttachComponent<RenderView>(returnedEntity);

	return returnedEntity;
}
