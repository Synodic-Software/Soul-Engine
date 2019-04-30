#pragma once

#include "Render/Raster/RenderCommands.h"

#include <boost/lockfree/queue.hpp>

class RasterModule;

class CommandList{

public:

	CommandList(std::shared_ptr<RasterModule>&);
	~CommandList() = default;

	CommandList(const CommandList&) = default;
	CommandList(CommandList &&) noexcept = default;

	CommandList& operator=(const CommandList&) = default;
	CommandList& operator=(CommandList &&) noexcept = default;

	// Agnostic raster API interface
	void Draw(DrawCommand&);
	void DrawIndirect(DrawIndirectCommand&);
	void UpdateBuffer(UpdateBufferCommand&);
	void UpdateTexture(UpdateTextureCommand&);
	void CopyBuffer(CopyBufferCommand&);
	void CopyTexture(CopyTextureCommand&);


private:

	std::shared_ptr<RasterModule> rasterModule_;

};