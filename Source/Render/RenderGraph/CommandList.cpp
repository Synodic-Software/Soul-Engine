#include "CommandList.h"
#include "Core/Utility/Exception/Exception.h"

CommandList::CommandList() : 
	commandList_(5)
{
}

void CommandList::Draw(DrawCommand&)
{

}

void CommandList::DrawIndirect(DrawIndirectCommand&)
{
	throw NotImplemented();
}

void CommandList::UpdateBuffer(UpdateBufferCommand&)
{
	throw NotImplemented();
}

void CommandList::UpdateTexture(UpdateTextureCommand&)
{
	throw NotImplemented();
}

void CommandList::CopyBuffer(CopyBufferCommand&)
{
	throw NotImplemented();
}

void CommandList::CopyTexture(CopyTextureCommand&)
{
	throw NotImplemented();
}