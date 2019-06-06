#pragma once

#include "Types.h"
#include "Core/Composition/Entity/Entity.h"

class RenderCommand{

public:

	RenderCommand() = default;
	~RenderCommand() = default;


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
	Entity vertexBuffer;
	Entity indexBuffer;

};

struct DrawIndirectCommand : RenderCommand {

};

struct UpdateBufferCommand : RenderCommand {

	uint offset;
	nonstd::span<std::byte> data;

};

struct UpdateTextureCommand : RenderCommand {

};

struct CopyBufferCommand : RenderCommand {

};

struct CopyTextureCommand : RenderCommand {

};