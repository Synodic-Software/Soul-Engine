#include "RasterManager.h"

#include "Platform/Platform.h"
#include "Graphics API/Vulkan/VulkanSwapChain.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Graphics API/RasterContext.h"

RasterManager::RasterManager(Scheduler& scheduler, EntityManager& entityManager) {

	if constexpr (Platform::IsDesktop()) {
		rasterContextVariant_.emplace<VulkanContext>(scheduler, entityManager);
		rasterContext_ = &std::get<VulkanContext>(rasterContextVariant_);
	}

}

void RasterManager::PreRaster() {

}

void RasterManager::Raster(Entity swapChain) {

	rasterContext_->Raster(swapChain);

}

void RasterManager::PostRaster() {

}

Entity RasterManager::CreateSurface(std::any& windowContext) const {

	if constexpr (Platform::IsDesktop()) {
		return rasterContext_->CreateSurface(windowContext);
	}

}

void RasterManager::ResizeSwapChain(Entity swapChain, int x, int y) {

	return rasterContext_->ResizeSwapChain(swapChain, x, y);

}

Entity RasterManager::CreateSwapChain(Entity device, Entity surface, glm::uvec2& size) const {

	return rasterContext_->CreateSwapChain(device, surface, size);

}

Entity RasterManager::CreateDevice(Entity surface) const {

	return rasterContext_->CreateDevice(surface);

}

void RasterManager::Synchronize() const{

	rasterContext_->Synchronize();

}
