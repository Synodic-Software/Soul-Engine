#pragma once

#include "Rasterer/RasterDevice.h"

class VulkanDevice final: public RasterDevice {

public:

	VulkanDevice() = default;
	~VulkanDevice() override = default;

	VulkanDevice(const VulkanDevice &) = delete;
	VulkanDevice(VulkanDevice &&) noexcept = default;

	VulkanDevice& operator=(const VulkanDevice &) = delete;
	VulkanDevice& operator=(VulkanDevice &&) noexcept = default;

};