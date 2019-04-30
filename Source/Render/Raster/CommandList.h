#pragma once

#include "Render/Raster/RenderCommands.h"

#include <boost/lockfree/queue.hpp>

class RasterModule;

class CommandListBase {
	CommandListBase(std::shared_ptr<RasterModule>&);
	~CommandListBase() = default;

	CommandListBase(const CommandListBase&) = default;
	CommandListBase(CommandListBase&&) noexcept = default;

	CommandListBase& operator=(const CommandListBase&) = default;
	CommandListBase& operator=(CommandListBase&&) noexcept = default;

	virtual void Start() = 0;
	virtual void End() = 0;

	// Agnostic raster API interface
	virtual void Draw(DrawCommand&) = 0;
	virtual void DrawIndirect(DrawIndirectCommand&) = 0;
	virtual void UpdateBuffer(UpdateBufferCommand&) = 0;
	virtual void UpdateTexture(UpdateTextureCommand&) = 0;
	virtual void CopyBuffer(CopyBufferCommand&) = 0;
	virtual void CopyTexture(CopyTextureCommand&) = 0;

};


//The public CommandList
class CommandList : public CommandListBase {

public:

	CommandList(std::shared_ptr<RasterModule>&);
	~CommandList() = default;

	CommandList(const CommandList&) = default;
	CommandList(CommandList &&) noexcept = default;

	CommandList& operator=(const CommandList&) = default;
	CommandList& operator=(CommandList &&) noexcept = default;

	void Start() override;
	void End() override;

	// Agnostic raster API interface
	void Draw(DrawCommand&) override;
	void DrawIndirect(DrawIndirectCommand&) override;
	void UpdateBuffer(UpdateBufferCommand&) override;
	void UpdateTexture(UpdateTextureCommand&) override;
	void CopyBuffer(CopyBufferCommand&) override;
	void CopyTexture(CopyTextureCommand&) override;


private:

	std::shared_ptr<RasterModule> rasterModule_;

};