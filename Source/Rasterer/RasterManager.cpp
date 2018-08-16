#include "RasterManager.h"

#include "Platform/Platform.h"
#include "Graphics API/Vulkan/VulkanSwapChain.h"
#include "Parallelism/Fiber/Scheduler.h"

RasterManager::RasterManager(Scheduler& scheduler, EntityManager& entityManager) {

	if constexpr (Platform::IsDesktop()) {
		rasterContextVariant_.emplace<VulkanContext>(scheduler,entityManager);
		rasterContext_ = &std::get<VulkanContext>(rasterContextVariant_);
	}

}

void RasterManager::PreRaster() {
	
}

void RasterManager::Raster() {
	
}

void RasterManager::PostRaster() {
	
}

Entity RasterManager::CreateSurface(std::any& windowContext) const {
	if constexpr (Platform::IsDesktop()) {
		return rasterContext_->CreateSurface(windowContext);
	}
}


std::unique_ptr<SwapChain> RasterManager::CreateSwapChain(Entity device, Entity surface, glm::uvec2& size) const{
	if constexpr (Platform::IsDesktop()) {
		return rasterContext_->CreateSwapChain(device, surface, size);
	}
}

Entity RasterManager::CreateDevice(Entity surface) const{
	if constexpr (Platform::IsDesktop()) {
		return rasterContext_->CreateDevice(surface);
	}
}
