#pragma once

#include "SwapChain.h"
#include "Composition/Entity/Entity.h"

#include <memory>
#include <any>

class RasterDevice;
class Surface;

enum class RasterAPI { NO_API, VULKAN }; // Backend types.

//Rendering Hardware Interface
class RasterContext {

public:

	RasterContext(RasterAPI);
	virtual ~RasterContext() = default;

	RasterContext(const RasterContext &) = delete;
	RasterContext(RasterContext &&) noexcept = default;

	RasterContext& operator=(const RasterContext &) = delete;
	RasterContext& operator=(RasterContext &&) noexcept = default;

	//All entity creation must also be manually deleted inside the Context
	virtual Entity CreateSurface(std::any&) = 0;
	virtual std::unique_ptr<SwapChain> CreateSwapChain(Entity, Entity, glm::uvec2&) = 0;
	virtual Entity CreateDevice(Entity) = 0;

protected:

	RasterAPI rasterAPI_;

};
