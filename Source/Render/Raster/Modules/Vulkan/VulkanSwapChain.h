#pragma once

#include "Pipeline/VulkanPipeline.h"
#include "VulkanFrameBuffer.h"
#include "Core/Composition/Component/Component.h"

#include <vulkan/vulkan.hpp>

class VulkanDevice;
class VulkanSurface;

class VulkanSwapChain : Component {

public:

	VulkanSwapChain(VulkanDevice&,
		VulkanSurface&, 
		bool,
		VulkanSwapChain* = nullptr);
	~VulkanSwapChain();

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = default;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = default;

	nonstd::span<vk::Image> Images();
	nonstd::span<vk::ImageView> ImageViews();
	[[nodiscard]] uint ActiveImageIndex() const;

	void AcquireImage(const vk::Semaphore&);

	[[nodiscard]] const vk::Device& Device() const;
	[[nodiscard]] vk::Extent2D Size() const;
	[[nodiscard]] vk::SwapchainKHR Handle() const;
	[[nodiscard]] vk::Semaphore RenderSemaphore() const;

private:

	vk::Device device_;

	std::vector<vk::Image> renderImages_;
	std::vector<vk::ImageView> renderImageViews_;

	std::vector<vk::Fence> imageFences_;
	std::vector<vk::Semaphore> presentSemaphores_;
	std::vector<vk::Semaphore> renderSemaphores_;
	uint activeImageIndex_;

	vk::Extent2D size_;
	vk::SwapchainKHR swapChain_;


};
