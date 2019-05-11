#pragma once

#include "VulkanPipeline.h"
#include "VulkanFrameBuffer.h"
#include "Command/VulkanCommandBuffer.h"

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

	void AquireImage();
	void Present(const vk::Queue&, VulkanCommandBuffer&);


private:

	std::shared_ptr<VulkanDevice> vkDevice_;

	vk::SwapchainKHR swapChain_;
	std::vector<SwapChainImage> images_;

	uint currentFrame_;
	uint activeImageIndex_;
	uint frameMax_;
	glm::uvec2 size_;

	std::vector<vk::Fence> frameFences_;
	std::vector<vk::Semaphore> presentSemaphores_;
	std::vector<vk::Semaphore> renderSemaphores_;


};