#pragma once

#include "Types.h"
#include "Render/Raster/RenderResource.h"

class RenderCommand{

public:

	RenderCommand() = default;
	~RenderCommand() = default;

	RenderCommand(const RenderCommand&) = default;
	RenderCommand(RenderCommand &&) noexcept = default;

	RenderCommand& operator=(const RenderCommand&) = default;
	RenderCommand& operator=(RenderCommand &&) noexcept = default;


};

struct DrawCommand : RenderCommand {

	//draw
	uint elementSize;
	uint indexOffset;
	uint vertexOffset;

	//scissor
	glm::uvec2 scissorOffset;
	glm::uvec2 scissorExtent;

	//data
	VertexBuffer vertexBuffer;
	IndexBuffer indexBuffer;

};

struct DrawIndirectCommand : RenderCommand {

};

struct UpdateBufferCommand : RenderCommand {

	uint size;
	uint offset;

};

struct UpdateTextureCommand : RenderCommand {

};

struct CopyBufferCommand : RenderCommand {

};

struct CopyTextureCommand : RenderCommand {

};