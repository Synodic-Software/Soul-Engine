#include "MockRasterBackend.h"

#include "Core/Utility/Exception/Exception.h"


void MockRasterBackend::Render()
{

	throw NotImplemented();

}

void MockRasterBackend::RenderPass(std::function<void()>)
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

void MockRasterBackend::Draw(DrawCommand& command)
{

	throw NotImplemented();

}

void MockRasterBackend::DrawIndirect(DrawIndirectCommand&)
{

	throw NotImplemented();

}
void MockRasterBackend::UpdateBuffer(UpdateBufferCommand&)
{

	throw NotImplemented();

}
void MockRasterBackend::UpdateTexture(UpdateTextureCommand&)
{

	throw NotImplemented();

}
void MockRasterBackend::CopyBuffer(CopyBufferCommand&)
{

	throw NotImplemented();

}
void MockRasterBackend::CopyTexture(CopyTextureCommand&)
{

	throw NotImplemented();

}