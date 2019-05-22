#include "RenderGraphBuilder.h"

RenderGraphBuilder::RenderGraphBuilder(std::shared_ptr<EntityRegistry>& entityRegistry):
	entityRegistry_(entityRegistry)
{
}

void RenderGraphBuilder::CreateOutput(RenderGraphOutputParameters&)
{
}

void RenderGraphBuilder::CreateInput(RenderGraphInputParameters&)
{
}

Entity RenderGraphBuilder::CreateGroup(ResourceGroupType type)
{

	Entity entity = entityRegistry_->CreateEntity();
	return entity;

}