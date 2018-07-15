#include "RasterManager.h"

#include "Platform/Platform.h"
#include "Graphics API/Vulkan/VulkanAPI.h"
#include "Graphics API/Vulkan/VulkanSwapChain.h"

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

std::unique_ptr<SwapChain> RasterManager::CreateSwapChain(std::any& windowContext, glm::uvec2& size) const{
	if constexpr (Platform::IsDesktop()) {
		return rasterAPI_->CreateSwapChain(windowContext, size);
	}
}
