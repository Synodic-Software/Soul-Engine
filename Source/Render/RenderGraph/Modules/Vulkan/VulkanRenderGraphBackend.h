#pragma once

#include "Render/RenderGraph/RenderGraphModule.h"


class VulkanRenderGraphBackend final : public RenderGraphModule {

public:

	VulkanRenderGraphBackend(std::shared_ptr<RasterModule>&);
	~VulkanRenderGraphBackend() override = default;

	VulkanRenderGraphBackend(const VulkanRenderGraphBackend &) = delete;
	VulkanRenderGraphBackend(VulkanRenderGraphBackend &&) noexcept = default;

	VulkanRenderGraphBackend& operator=(const VulkanRenderGraphBackend &) = delete;
	VulkanRenderGraphBackend& operator=(VulkanRenderGraphBackend &&) noexcept = default;


	void CreatePass(std::string, std::function<std::function<void(CommandList&)>()>) override;


};