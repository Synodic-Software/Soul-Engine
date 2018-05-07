#include "RasterManager.h"
#include "Graphics API/Vulkan/VulkanAPI.h"

RasterManager::RasterManager() {
	rasterAPI.reset(new VulkanAPI());
}

void RasterManager::PreRaster() {
	
}

void RasterManager::Raster() {
	
}

void RasterManager::PostRaster() {
	
}