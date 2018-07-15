#pragma once

#include "Rasterer/Graphics API/SwapChain.h"

#include <vulkan/vulkan.hpp>

#include <any>

struct SwapChainImage {
	vk::Image image;
	vk::ImageView view;
	vk::Fence fence;
};

class VulkanSwapChain : public SwapChain {

public:

	VulkanSwapChain(std::shared_ptr<vk::Instance>&, std::vector<vk::PhysicalDevice>&, std::vector<char const*>&, std::any&, glm::uvec2&);
	~VulkanSwapChain() override;

	VulkanSwapChain(const VulkanSwapChain&) = delete;
	VulkanSwapChain(VulkanSwapChain&& o) noexcept = delete;

	VulkanSwapChain& operator=(const VulkanSwapChain&) = delete;
	VulkanSwapChain& operator=(VulkanSwapChain&& other) noexcept = delete;

	void Resize(glm::uvec2) override;


private:

	std::shared_ptr<vk::Instance> vulkanInstance_;

	//TODO remove hardcoded device creation
	vk::Device logicalDevice_;
	std::vector<SwapChainImage> images_;

	vk::SurfaceKHR surface_;
	vk::SwapchainKHR swapChain_;

	vk::ColorSpaceKHR colorSpace_;

	bool vSync;

};
