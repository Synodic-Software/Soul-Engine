#pragma once

#include "Render/Raster/RenderCommands.h"

#include "Core/Utility/Thread/ThreadLocal.h"

class CommandList {

public:

	// Agnostic raster API interface
	void Draw(DrawCommand&);
	void DrawIndirect(DrawIndirectCommand&);
	void UpdateBuffer(UpdateBufferCommand&);
	void UpdateTexture(UpdateTextureCommand&);
	void CopyBuffer(CopyBufferCommand&);
	void CopyTexture(CopyTextureCommand&);


private:

	ThreadLocal<std::vector<DrawCommand>> drawList_;
	ThreadLocal<std::vector<DrawIndirectCommand>> drawIndirectList_;
	ThreadLocal<std::vector<UpdateBufferCommand>> updateBufferList_;
	ThreadLocal<std::vector<UpdateTextureCommand>> updateTextureList_;
	ThreadLocal<std::vector<CopyBufferCommand>> copyBufferList_;
	ThreadLocal<std::vector<CopyTextureCommand>> copyTextureList_;

};