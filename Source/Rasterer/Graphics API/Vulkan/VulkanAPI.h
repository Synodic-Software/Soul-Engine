#pragma once

#include "Rasterer/Graphics API/GraphicsAPI.h"

#include "vulkan/vulkan.hpp"

class VulkanAPI final: public GraphicsAPI {

public:

	VulkanAPI();
	~VulkanAPI() override;

	VulkanAPI(const VulkanAPI&) = delete;
	VulkanAPI(VulkanAPI&&) noexcept = delete;

	VulkanAPI& operator=(const VulkanAPI&) = delete;
	VulkanAPI& operator=(VulkanAPI&&) noexcept = delete;

	std::unique_ptr<SwapChain> CreateSwapChain(std::any&, glm::uvec2&) override;

private:

	std::shared_ptr<vk::Instance> vulkanInstance_;

	//TODO alternate storage
	std::vector<char const*> requiredExtensions_;


	std::vector<vk::PhysicalDevice> physicalDevices_;
	std::vector<vk::Device> logicalDevices_;

};