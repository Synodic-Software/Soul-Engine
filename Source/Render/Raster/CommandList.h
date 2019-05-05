#pragma once

#include "Render/Raster/RenderCommands.h"

#include "Core/Utility/Thread/ThreadLocal.h"

#include <memory>

class RasterModule;

class CommandList {

public:

	CommandList(std::shared_ptr<RasterModule>&);

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