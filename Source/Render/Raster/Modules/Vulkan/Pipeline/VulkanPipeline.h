#pragma once

#include "Render/Raster/Modules/Vulkan/VulkanRenderPass.h"
#include "Render/Raster/Modules/Vulkan/VulkanShader.h"
#include "Render/Raster/Modules/Vulkan/Buffer/VulkanBuffer.h"
#include "VulkanPipelineLayout.h"

#include <vulkan/vulkan.hpp>

class Resource;
class VulkanDevice;
class Vertex;

class VulkanPipeline {

public:

	VulkanPipeline(const vk::Device&, vk::Extent2D&, const Resource&, const Resource&, vk::Format);
	~VulkanPipeline();

	VulkanPipeline(const VulkanPipeline&) = delete;
	VulkanPipeline(VulkanPipeline&&) noexcept = delete;

	VulkanPipeline& operator=(const VulkanPipeline&) = delete;
	VulkanPipeline& operator=(VulkanPipeline&&) noexcept = delete;

	const vk::Pipeline& GetPipeline() const;


private:

	vk::Device device_;

	std::vector<VulkanShader> stages_;

	VulkanPipelineLayout pipelineLayout_;
	vk::Pipeline pipeline_;
	vk::PipelineCache pipelineCache_;


};
