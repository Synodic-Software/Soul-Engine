#pragma once

enum class RasterAPI { NOAPI, VULKAN }; // Backend types.

//Rendering Hardware Interface
class GraphicsAPI {

public:

	GraphicsAPI(RasterAPI);
	~GraphicsAPI() = default;

	GraphicsAPI(const GraphicsAPI &) = delete;
	GraphicsAPI(GraphicsAPI &&) noexcept = default;

	GraphicsAPI& operator=(const GraphicsAPI &) = delete;
	GraphicsAPI& operator=(GraphicsAPI &&) noexcept = default;

protected:

	RasterAPI rasterAPI_;

};
