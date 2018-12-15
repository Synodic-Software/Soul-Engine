#pragma once

#include "Rasterer/Graphics API/SwapChain.h"
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

class VulkanSwapChain : public SwapChain, Component<VulkanSwapChain> {

public:

	VulkanSwapChain(EntityManager*, Entity, Entity, glm::uvec2&);
	~VulkanSwapChain() override = default;

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = default;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = default;

	void Terminate() override;
//	void Cleanup(); // TODO: Move this up

	void Resize(Entity surface, glm::uvec2 size) override;
	void Draw() override;

private:

	EntityManager* entityManager_;

	Entity device_;

	std::unique_ptr<VulkanPipeline> pipeline_;
	std::vector<SwapChainImage> images_;
	std::vector<VulkanFramebuffer> frameBuffers_;
	std::vector<vk::CommandBuffer> commandBuffers_;

	vk::SwapchainKHR swapChain_;
	vk::ColorSpaceKHR colorSpace_;

	//TODO place into VulkanContext
	std::vector<vk::Semaphore> imageAvailableSemaphores;
	std::vector<vk::Semaphore> renderFinishedSemaphores;
	std::vector<vk::Fence> inFlightFences;
	size_t currentFrame;
	uint flightFramesCount;

	bool vSync;

	void BuildSwapChain(Entity surface, glm::uvec2& size, bool createPipeline);
};
