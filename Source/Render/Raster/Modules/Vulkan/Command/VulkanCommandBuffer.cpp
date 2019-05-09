#include "VulkanCommandBuffer.h"

#include "VulkanCommandPool.h"

#include "Render/Raster/Modules/Vulkan/VulkanDevice.h"

VulkanCommandBuffer::VulkanCommandBuffer(std::shared_ptr<VulkanCommandPool>& commandPool,
	const std::shared_ptr<VulkanDevice>& vulkanDevice,
	vk::CommandBufferUsageFlagBits usage,
	vk::CommandBufferLevel bufferLevel):
	commandPool_(commandPool),
	device_(vulkanDevice), usage_(usage)
{

	const vk::Device& logicalDevice = device_->GetLogical();

	vk::CommandBufferAllocateInfo allocInfo;
	allocInfo.level = bufferLevel;
	allocInfo.commandPool = commandPool_->GetCommandPool();
	allocInfo.commandBufferCount = 1;

	commandBuffer_ = logicalDevice.allocateCommandBuffers(allocInfo).front();

}

VulkanCommandBuffer::~VulkanCommandBuffer()
{

	const vk::Device& logicalDevice = device_->GetLogical();

	logicalDevice.freeCommandBuffers(commandPool_->GetCommandPool(), commandBuffer_);

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

void VulkanCommandBuffer::Submit()
{

	vk::SubmitInfo submitInfo;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer_;

	device_->GetGraphicsQueue().submit(submitInfo, nullptr);
	device_->GetGraphicsQueue().waitIdle();

}

const vk::CommandBuffer& VulkanCommandBuffer::Get() const
{

	return commandBuffer_;

}