#pragma once

#include "Render/Raster/RenderCommands.h"

#include <boost/lockfree/queue.hpp>

class CommandList{

public:

	CommandList();
	~CommandList() = default;

	CommandList(const CommandList&) = default;
	CommandList(CommandList &&) noexcept = default;

	CommandList& operator=(const CommandList&) = default;
	CommandList& operator=(CommandList &&) noexcept = default;

	// Agnostic raster API interface
	virtual void Draw(DrawCommand&) = 0;
	virtual void DrawIndirect(DrawIndirectCommand&) = 0;
	virtual void UpdateBuffer(UpdateBufferCommand&) = 0;
	virtual void UpdateTexture(UpdateTextureCommand&) = 0;
	virtual void CopyBuffer(CopyBufferCommand&) = 0;
	virtual void CopyTexture(CopyTextureCommand&) = 0;

private:

	boost::lockfree::queue<RenderCommand> commandList_;

};