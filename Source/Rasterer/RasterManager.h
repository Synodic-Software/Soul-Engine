#pragma once

#include "Graphics API/RasterContext.h"
#include "Graphics API/Vulkan/VulkanContext.h"
#include "Graphics API/SwapChain.h"

#include <memory>
#include <variant>
#include <any>

class RasterManager {

public:

	RasterManager(EntityManager&);
	~RasterManager() = default;

	RasterManager(const RasterManager&) = delete;
	RasterManager(RasterManager&& o) noexcept = delete;

	RasterManager& operator=(const RasterManager&) = delete;
	RasterManager& operator=(RasterManager&& other) noexcept = delete;

	//Draw and Update steps called from the main loop
	void PreRaster();
	void Raster();
	void PostRaster();

	Entity CreateSurface(std::any& windowContext) const;
	std::unique_ptr<SwapChain> CreateSwapChain(Entity, Entity, glm::uvec2&) const;
	Entity CreateDevice(Entity) const;


private:

	std::variant<std::monostate, VulkanContext> rasterContextVariant_;
	RasterContext* rasterContext_;

};
