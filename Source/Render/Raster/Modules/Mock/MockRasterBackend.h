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
	Entity CreatePass(Entity) override;
	Entity CreateSubPass(Entity) override;
	void ExecutePass(Entity, CommandList&) override;

	Entity RegisterSurface(std::any, glm::uvec2) override;
	void UpdateSurface(Entity, glm::uvec2) override;
	void RemoveSurface(Entity) override;

	void Compile(CommandList&) override;

};
