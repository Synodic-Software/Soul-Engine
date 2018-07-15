#pragma once

#include "Graphics API/GraphicsAPI.h"
#include "Graphics API/Vulkan/VulkanAPI.h"
#include "Graphics API/SwapChain.h"

#include <memory>
#include <variant>
#include <any>

class RasterManager {

public:

	RasterManager();
	~RasterManager() = default;

	RasterManager(const RasterManager&) = delete;
	RasterManager(RasterManager&& o) noexcept = delete;

	RasterManager& operator=(const RasterManager&) = delete;
	RasterManager& operator=(RasterManager&& other) noexcept = delete;

	//Draw and Update steps called from the main loop
	void PreRaster();
	void Raster();
	void PostRaster();

	std::unique_ptr<SwapChain> CreateSwapChain(std::any&, glm::uvec2&) const;


private:

	std::variant<std::monostate, VulkanAPI> rasterAPIVariant_;
	GraphicsAPI* rasterAPI_;

};
