#pragma once

#include "Rasterer/Graphics API/GraphicsAPI.h"

#include "vulkan/vulkan.hpp"

class VulkanAPI : public GraphicsAPI {

public:

	VulkanAPI();
	~VulkanAPI() = default;

	VulkanAPI(const VulkanAPI&) = delete;
	VulkanAPI(VulkanAPI&&) noexcept = delete;

	VulkanAPI& operator=(const VulkanAPI&) = delete;
	VulkanAPI& operator=(VulkanAPI&&) noexcept = delete;

private:

	vk::UniqueInstance vulkanInstance_;

};