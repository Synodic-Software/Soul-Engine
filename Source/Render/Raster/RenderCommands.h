#pragma once

#include "Types.h"
#include "Core/Composition/Entity/Entity.h"
#include "RenderResource.h"
#include "Core/Structure/External/ExternalBuffer.h"

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

	UpdateBufferCommand(BufferType typeIn): type(typeIn)
	{
	}

	uint offset;
	ExternalBuffer<std::byte> data;
	Entity buffer;

private:

	BufferType type;


};

struct UpdateTextureCommand : RenderCommand {

};

struct CopyBufferCommand : RenderCommand {

};

struct CopyTextureCommand : RenderCommand {

};