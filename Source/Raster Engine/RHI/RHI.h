#pragma once

enum RasterAPI { Vulkan }; // Backend types.

//Rendering Hardware Interface
class RHI {

public:

	RHI();
	~RHI();


private:

	RasterAPI backendType;

};
