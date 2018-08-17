#pragma once

#include "Rasterer/Graphics API/SwapChain.h"
#include "Composition/Entity/EntityManager.h"
#include "VulkanPipeline.h"
#include "VulkanFramebuffer.h"

#include <vulkan/vulkan.hpp>

class VulkanSurface;

struct SwapChainImage {
	vk::Image image;
	vk::ImageView view;
	vk::Fence fence;
};

class VulkanSwapChain : public SwapChain {

public:

	VulkanSwapChain(EntityManager&, Entity, Entity, glm::uvec2&);
	~VulkanSwapChain() override;

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = delete;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = delete;

	void Resize(glm::uvec2) override;
	void Draw() override;

private:

	EntityManager& entityManager_;

	Entity device_;

	std::unique_ptr<VulkanPipeline> pipeline_;
	std::vector<SwapChainImage> images_;
	std::vector<VulkanFramebuffer> frameBuffers_;
	std::vector<vk::CommandBuffer> commandBuffers_;

	vk::SwapchainKHR swapChain_;
	vk::ColorSpaceKHR colorSpace_;

	std::vector<vk::Semaphore> imageAvailableSemaphores;
	std::vector<vk::Semaphore> renderFinishedSemaphores;
	std::vector<vk::Fence> inFlightFences;
	size_t currentFrame;
	uint flightFramesCount;

	bool vSync;

};