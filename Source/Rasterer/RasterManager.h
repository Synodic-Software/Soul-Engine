#pragma once

class RasterContext;
#include "Graphics API/Vulkan/VulkanContext.h"
#include "Graphics API/SwapChain.h"

#include <memory>
#include <variant>
#include <any>

class Scheduler;

class RasterManager {

public:

	RasterManager(Scheduler&, EntityManager&);
	~RasterManager() = default;

	RasterManager(const RasterManager&) = delete;
	RasterManager(RasterManager&& o) noexcept = delete;

	RasterManager& operator=(const RasterManager&) = delete;
	RasterManager& operator=(RasterManager&& other) noexcept = delete;

	//Draw and Update steps called from the main loop
	void PreRaster();
	void Raster(Entity);
	void PostRaster();

	void ResizeSwapChain(Entity, Entity, int, int);

	Entity CreateSurface(std::any& windowContext) const;
	Entity CreateSwapChain(Entity, Entity, glm::uvec2&) const;
	Entity CreateDevice(Entity) const;

	void Synchronize() const;

private:

	std::variant<std::monostate, VulkanContext> rasterContextVariant_;
	RasterContext* rasterContext_;

};
