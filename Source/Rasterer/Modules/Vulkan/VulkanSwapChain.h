#pragma once

#include "VulkanPipeline.h"
#include "VulkanFrameBuffer.h"

#include <vulkan/vulkan.hpp>
#include <glm/vec2.hpp>

class VulkanDevice;

struct SwapChainImage {
	vk::Image image;
	vk::ImageView view;
	vk::Fence fence;
};

class VulkanSwapChain {

public:

	VulkanSwapChain(std::shared_ptr<VulkanDevice>&, vk::SurfaceKHR&, vk::Format, vk::ColorSpaceKHR, glm::uvec2&, bool, VulkanSwapChain* = nullptr);
	~VulkanSwapChain();

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = default;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = default;

	void Present();


private:

	std::shared_ptr<VulkanDevice> vkDevice_;

	vk::SwapchainKHR swapChain_;
	std::vector<SwapChainImage> images_;

	glm::uvec2 size_;

	//TODO: refactor
	std::vector<vk::Semaphore> imageAvailableSemaphores;
	std::vector<vk::Semaphore> renderFinishedSemaphores;
	std::vector<vk::Fence> inFlightFences;
	size_t currentFrame;
	uint flightFramesCount;

	//TODO: refactor
	std::vector<VulkanFrameBuffer> frameBuffers_;
	std::vector<vk::CommandBuffer> commandBuffers_;
	std::unique_ptr<VulkanPipeline> pipeline_;

};
