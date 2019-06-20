#include "VulkanCommandBuffer.h"

#include "VulkanCommandPool.h"

#include "Render/Raster/Modules/Vulkan/Device/VulkanDevice.h"

VulkanCommandBuffer::VulkanCommandBuffer(std::shared_ptr<VulkanCommandPool>& commandPool,
	const vk::Device& vulkanDevice,
	vk::CommandBufferUsageFlagBits usage,
	vk::CommandBufferLevel bufferLevel):
	commandPool_(commandPool),
	device_(vulkanDevice), usage_(usage)
{

	vk::CommandBufferAllocateInfo allocInfo;
	allocInfo.level = bufferLevel;
	allocInfo.commandPool = commandPool_->GetCommandPool();
	allocInfo.commandBufferCount = 1;

	commandBuffer_ = device_.allocateCommandBuffers(allocInfo).front();

}

VulkanCommandBuffer::~VulkanCommandBuffer()
{

	device_.freeCommandBuffers(commandPool_->GetCommandPool(), commandBuffer_);

}


void VulkanCommandBuffer::Begin()
{

	vk::CommandBufferBeginInfo beginInfo;
	beginInfo.flags = usage_;

	commandBuffer_.begin(beginInfo);

}

void VulkanCommandBuffer::End()
{

	commandBuffer_.end();

}

const vk::CommandBuffer& VulkanCommandBuffer::Get() const
{

	return commandBuffer_;

}