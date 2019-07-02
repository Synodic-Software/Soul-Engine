#pragma once

#include <vulkan/vulkan.hpp>

struct SurfaceFormat {

	vk::ColorSpaceKHR colorSpace;
	vk::Format colorFormat;

};

class VulkanSurface {

public:

	VulkanSurface(const vk::Instance&, const vk::SurfaceKHR&);
	~VulkanSurface();

	VulkanSurface(const VulkanSurface&) = delete;
	VulkanSurface(VulkanSurface&&) noexcept = default;

	VulkanSurface& operator=(const VulkanSurface&) = delete;
	VulkanSurface& operator=(VulkanSurface&&) noexcept = default;

	vk::SurfaceKHR Handle();

	SurfaceFormat VulkanSurface::Format(const vk::PhysicalDevice&) const;

private:

	vk::Instance instance_;
	vk::SurfaceKHR surface_;
	vk::Extent2D size_;

};
