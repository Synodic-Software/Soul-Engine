#include "CommandList.h"
#include "Core/Utility/Exception/Exception.h"

#include "Render/Raster/RasterModule.h"

void CommandList::Draw(DrawCommand& command)
{

	drawList_->push_back(command);

}

void CommandList::DrawIndirect(DrawIndirectCommand& command)
{

	drawIndirectList_->push_back(command);

}

void CommandList::UpdateBuffer(UpdateBufferCommand& command)
{

	updateBufferList_->push_back(command);

}

void CommandList::UpdateTexture(UpdateTextureCommand& command)
{

	updateTextureList_->push_back(command);

}

void CommandList::CopyBuffer(CopyBufferCommand& command)
{

	copyBufferList_->push_back(command);

}

void CommandList::CopyTexture(CopyTextureCommand& command)
{

	copyTextureList_->push_back(command);

}