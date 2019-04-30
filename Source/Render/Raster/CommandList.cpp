#include "CommandList.h"
#include "Core/Utility/Exception/Exception.h"

#include "Render/Raster/RasterModule.h"

CommandList::CommandList(std::shared_ptr<RasterModule>& rasterModule): 
	rasterModule_(rasterModule)
{
}

void CommandList::Draw(DrawCommand& command)
{

	rasterModule_->Draw(command);

}

void CommandList::DrawIndirect(DrawIndirectCommand& command)
{
	rasterModule_->DrawIndirect(command);
}

void CommandList::UpdateBuffer(UpdateBufferCommand& command)
{
	rasterModule_->UpdateBuffer(command);
}

void CommandList::UpdateTexture(UpdateTextureCommand& command)
{
	rasterModule_->UpdateTexture(command);
}

void CommandList::CopyBuffer(CopyBufferCommand& command)
{
	rasterModule_->CopyBuffer(command);
}

void CommandList::CopyTexture(CopyTextureCommand& command)
{
	rasterModule_->CopyTexture(command);
}