#pragma once

#include "VulkanPipeline.h"
#include "VulkanFramebuffer.h"
#include "Composition/Component/Component.h"

#include <vulkan/vulkan.hpp>

class VulkanSurface;
class EntityManager;

struct SwapChainImage {
	vk::Image image;
	vk::ImageView view;
	vk::Fence fence;
};

class VulkanSwapChain : Component<VulkanSwapChain> {

public:

	VulkanSwapChain(EntityManager*, Entity, Entity, glm::uvec2&);
	~VulkanSwapChain() override;

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = default;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = default;

	void Resize(Entity surface, glm::uvec2 size);
	void Draw();

private:

	EntityManager* entityManager_;

	Entity device_;

	std::unique_ptr<VulkanPipeline> pipeline_;
	std::vector<SwapChainImage> images_;
	std::vector<VulkanFramebuffer> frameBuffers_;
	std::vector<vk::CommandBuffer> commandBuffers_;

	vk::SwapchainKHR swapChain_;
	vk::ColorSpaceKHR colorSpace_;

	//TODO place into VulkanRasterBackend
	std::vector<vk::Semaphore> imageAvailableSemaphores;
	std::vector<vk::Semaphore> renderFinishedSemaphores;
	std::vector<vk::Fence> inFlightFences;
	size_t currentFrame;
	uint flightFramesCount;

	bool vSync;

	void BuildSwapChain(Entity surface, glm::uvec2& size, bool createPipeline);
};
