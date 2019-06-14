#pragma once

#include "Pipeline/VulkanPipeline.h"
#include "VulkanFrameBuffer.h"
#include "Command/VulkanCommandBuffer.h"

#include <vulkan/vulkan.hpp>
#include <glm/vec2.hpp>

class VulkanDevice;

class VulkanSwapChain {

public:

	VulkanSwapChain(std::shared_ptr<VulkanDevice>&,
		vk::SurfaceKHR&,
		vk::ColorSpaceKHR,
		const vk::Extent2D&,
		bool,
		VulkanSwapChain* = nullptr);
	~VulkanSwapChain();

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = default;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = default;

	void AquireImage();
	void Present(VulkanCommandBuffer&);

	vk::Format GetFormat();
	glm::uvec2 GetSize();

private:


	std::shared_ptr<VulkanDevice> vkDevice_;


	std::vector<vk::Image> renderBuffers_;
	std::vector<vk::ImageView> renderBufferViews_;

	uint currentFrame_;
	uint activeImageIndex_;
	uint frameMax_;

	vk::Extent2D size_;
	vk::SwapchainKHR swapChain_;

	std::vector<vk::Fence> frameFences_;
	std::vector<vk::Semaphore> presentSemaphores_;
	std::vector<vk::Semaphore> renderSemaphores_;


};
