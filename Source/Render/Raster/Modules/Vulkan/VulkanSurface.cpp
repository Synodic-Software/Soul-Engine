#include "VulkanSurface.h"

VulkanSurface::VulkanSurface(const vk::Instance& instance, const vk::SurfaceKHR& surface): 
	instance_(instance),
	surface_(surface), 
	format_ {vk::ColorSpaceKHR::eSrgbNonlinear, vk::Format::eB8G8R8A8Unorm}
{
}

VulkanSurface::~VulkanSurface()
{

	instance_.destroySurfaceKHR(surface_);

}

vk::SurfaceKHR VulkanSurface::Handle()
{

	return surface_;

}

SurfaceFormat VulkanSurface::UpdateFormat(const vk::PhysicalDevice& physicalDevice)
{

	const auto formats = physicalDevice.getSurfaceFormatsKHR(surface_);

	// TODO: pick formats better
	if (!formats.empty() && formats.front().format == vk::Format::eUndefined) {
		return format_;
	}

	for (const auto& format : formats) {

		if (format.format == vk::Format::eB8G8R8A8Unorm &&
			format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
			return format_;
		}
	}

	format_.colorFormat = formats.front().format;
	format_.colorSpace = formats.front().colorSpace;

	return format_;
}

SurfaceFormat VulkanSurface::Format() const
{

	return format_;

}