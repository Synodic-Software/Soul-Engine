#pragma once

#include "SwapChain.h"

#include <memory>
#include <any>

enum class RasterAPI { NOAPI, VULKAN }; // Backend types.

//Rendering Hardware Interface
class GraphicsAPI {

public:

	GraphicsAPI(RasterAPI);
	virtual ~GraphicsAPI() = default;

	GraphicsAPI(const GraphicsAPI &) = delete;
	GraphicsAPI(GraphicsAPI &&) noexcept = default;

	GraphicsAPI& operator=(const GraphicsAPI &) = delete;
	GraphicsAPI& operator=(GraphicsAPI &&) noexcept = default;

	//
	virtual std::unique_ptr<SwapChain> CreateSwapChain(std::any&, glm::uvec2&) = 0;

protected:

	RasterAPI rasterAPI_;

};
