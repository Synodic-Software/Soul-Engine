#pragma once

#include "Device/VulkanDevice.h"

#include <vulkan/vulkan.hpp>

class VulkanSurface {

public:

	VulkanSurface(const vk::Instance&, const vk::SurfaceKHR&);
	~VulkanSurface();

	VulkanSurface(const VulkanSurface&) = delete;
	VulkanSurface(VulkanSurface&&) noexcept = default;

	VulkanSurface& operator=(const VulkanSurface&) = delete;
	VulkanSurface& operator=(VulkanSurface&&) noexcept = default;

	vk::SurfaceKHR Handle();

	vk::SurfaceFormatKHR UpdateFormat(const VulkanDevice& device);
	vk::SurfaceFormatKHR Format() const;

private:

	vk::Instance instance_;
	vk::SurfaceKHR surface_;
	vk::Extent2D size_;

	vk::SurfaceFormatKHR format_;

};
