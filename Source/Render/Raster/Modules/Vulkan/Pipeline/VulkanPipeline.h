#pragma once

#include "Render/Raster/Modules/Vulkan/VulkanRenderPass.h"
#include "Render/Raster/Modules/Vulkan/VulkanShader.h"
#include "Render/Raster/Modules/Vulkan/Buffer/VulkanBuffer.h"
#include "VulkanPipelineCache.h"
#include "VulkanPipelineLayout.h"

#include <vulkan/vulkan.hpp>

class Resource;
class VulkanDevice;
class Vertex;

class VulkanPipeline {

public:

	VulkanPipeline(const vk::Device&, vk::Extent2D&);
	~VulkanPipeline();

	VulkanPipeline(const VulkanPipeline&) = delete;
	VulkanPipeline(VulkanPipeline&&) noexcept = delete;

	VulkanPipeline& operator=(const VulkanPipeline&) = delete;
	VulkanPipeline& operator=(VulkanPipeline&&) noexcept = delete;

	const vk::Pipeline& GetPipeline() const;


private:

	vk::Device device_;

	std::vector<VulkanShader> stages_;

	VulkanPipelineCache pipelineCache_;
	VulkanPipelineLayout pipelineLayout_;

	vk::Pipeline pipeline_;


};
