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

void MockRasterBackend(uint, glm::uvec2)
{

	throw NotImplemented();

}

void MockRasterBackend::RemoveSurface(uint surface)
{

	throw NotImplemented();

}

void MockRasterBackend::Draw()
{

	throw NotImplemented();

}

void MockRasterBackend::DrawIndirect()
{

	throw NotImplemented();

}

void MockRasterBackend::UpdateBuffer()
{

	throw NotImplemented();

}

void MockRasterBackend::UpdateTexture()
{

	throw NotImplemented();

}

void MockRasterBackend::CopyBuffer()
{

	throw NotImplemented();

}

void MockRasterBackend::CopyTexture()
{

	throw NotImplemented();

}