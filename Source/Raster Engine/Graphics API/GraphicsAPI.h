#pragma once

enum RasterAPI { Vulkan }; // Backend types.

//Rendering Hardware Interface
class GraphicsAPI {

public:

	GraphicsAPI();
	~GraphicsAPI();


private:

	RasterAPI backendType;

};
