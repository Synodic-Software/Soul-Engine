#include "MockRasterBackend.h"

#include "Core/Utility/Exception/Exception.h"


void MockRasterBackend::Render()
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