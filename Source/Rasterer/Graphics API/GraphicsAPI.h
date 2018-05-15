#pragma once

enum RasterAPI { Vulkan, NoAPI }; // Backend types.

//Rendering Hardware Interface
class GraphicsAPI {

public:
    
	GraphicsAPI();


protected:

	RasterAPI backendType;

};
