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
