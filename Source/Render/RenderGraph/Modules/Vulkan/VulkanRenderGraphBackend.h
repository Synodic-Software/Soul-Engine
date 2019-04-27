#pragma once

#include "Render/RenderGraph/RenderGraphModule.h"


class VulkanRenderGraphBackend final : public RenderGraphModule {

public:

	VulkanRenderGraphBackend() = default;
	~VulkanRenderGraphBackend() override = default;

	VulkanRenderGraphBackend(const VulkanRenderGraphBackend &) = delete;
	VulkanRenderGraphBackend(VulkanRenderGraphBackend &&) noexcept = default;

	VulkanRenderGraphBackend& operator=(const VulkanRenderGraphBackend &) = delete;
	VulkanRenderGraphBackend& operator=(VulkanRenderGraphBackend &&) noexcept = default;


};