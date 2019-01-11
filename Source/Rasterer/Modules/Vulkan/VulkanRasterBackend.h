#pragma once

#include "Rasterer/RasterBackend.h"

class VulkanRasterBackend final: public RasterBackend {

public:

	VulkanRasterBackend() = default;
	~VulkanRasterBackend() override = default;

	VulkanRasterBackend(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend(VulkanRasterBackend &&) noexcept = default;

	VulkanRasterBackend& operator=(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend& operator=(VulkanRasterBackend &&) noexcept = default;

	void Draw() override;
	void DrawIndirect() override;

	std::shared_ptr<RasterDevice> CreateDevice() override;

};