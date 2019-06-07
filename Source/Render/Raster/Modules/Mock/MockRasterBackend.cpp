#include "MockRasterBackend.h"

#include "Core/Utility/Exception/Exception.h"


void MockRasterBackend::Render()
{

	throw NotImplemented();

}

void MockRasterBackend::RenderPass(std::function<CommandList()>)
{

	throw NotImplemented();

}

uint MockRasterBackend::RegisterSurface(std::any, glm::uvec2 size)
{

	throw NotImplemented();

}

void MockRasterBackend::UpdateSurface(uint, glm::uvec2)
{

	throw NotImplemented();

}

void MockRasterBackend::RemoveSurface(uint surface)
{

	throw NotImplemented();

}

void MockRasterBackend::CompileCommands(CommandList&)
{

	throw NotImplemented();

}

void MockRasterBackend::ExecuteCommands(CommandList&)
{

	throw NotImplemented();

}