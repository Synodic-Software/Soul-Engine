#pragma once

#include "Render/Raster/RenderCommands.h"

#include "Core/Utility/Thread/ThreadLocal.h"

#include <memory>

class RasterModule;

class CommandList {

public:

	CommandList(RasterModule*);

	// Agnostic raster API interface
	void Draw(DrawCommand&);
	void DrawIndirect(DrawIndirectCommand&);
	void UpdateBuffer(UpdateBufferCommand&);
	void UpdateTexture(UpdateTextureCommand&);
	void CopyBuffer(CopyBufferCommand&);
	void CopyTexture(CopyTextureCommand&);


private:

	RasterModule* rasterModule_;

};