#pragma once

#include "Render/Raster/RasterModule.h"

class MockRasterBackend;

class MockRasterBackend final : public RasterModule {

public:

	MockRasterBackend() = default;
	~MockRasterBackend() override = default;

	MockRasterBackend(const MockRasterBackend &) = delete;
	MockRasterBackend(MockRasterBackend &&) noexcept = default;

	MockRasterBackend& operator=(const MockRasterBackend &) = delete;
	MockRasterBackend& operator=(MockRasterBackend &&) noexcept = default;

	void Render() override;
	void RenderPass(std::function<void()>) override;

	uint RegisterSurface(std::any, glm::uvec2) override;
	void UpdateSurface(uint, glm::uvec2) override;
	void RemoveSurface(uint) override;


	// Agnostic raster API interface
	void Draw(DrawCommand&) override;
	void DrawIndirect(DrawIndirectCommand&) override;
	void UpdateBuffer(UpdateBufferCommand&) override;
	void UpdateTexture(UpdateTextureCommand&) override;
	void CopyBuffer(CopyBufferCommand&) override;
	void CopyTexture(CopyTextureCommand&) override;


};
