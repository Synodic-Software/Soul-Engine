#include "RasterManager.h"

#include "Platform/Platform.h"
#include "Graphics API/Vulkan/VulkanAPI.h"

RasterManager::RasterManager() {

	if constexpr (Platform::IsDesktop()) {
		rasterAPIVariant_.emplace<VulkanAPI>();
		rasterAPI_ = &std::get<VulkanAPI>(rasterAPIVariant_);
	}

}

void RasterManager::PreRaster() {
	
}

void RasterManager::Raster() {
	
}

void RasterManager::PostRaster() {
	
}
