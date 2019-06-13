#include "MockRasterBackend.h"

#include "Core/Utility/Exception/Exception.h"


void MockRasterBackend::Render()
{

	throw NotImplemented();

}

Entity MockRasterBackend::CreatePass(Entity)
{

	throw NotImplemented();

}

Entity MockRasterBackend::CreateSubPass(Entity)
{

	throw NotImplemented();

}

void MockRasterBackend::ExecutePass(Entity, CommandList&)
{

	throw NotImplemented();

}

Entity MockRasterBackend::RegisterSurface(std::any, glm::uvec2 size)
{

	throw NotImplemented();

}

void MockRasterBackend::UpdateSurface(Entity, glm::uvec2)
{

	throw NotImplemented();

}

void MockRasterBackend::RemoveSurface(Entity surface)
{

	throw NotImplemented();

}

void MockRasterBackend::Compile(CommandList&)
{

	throw NotImplemented();

}
