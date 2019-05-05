#pragma once

#include "Types.h"

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

};

struct DrawIndirectCommand : RenderCommand {

};

struct UpdateBufferCommand : RenderCommand {

};

struct UpdateTextureCommand : RenderCommand {

};

struct CopyBufferCommand : RenderCommand {

};

struct CopyTextureCommand : RenderCommand {

};