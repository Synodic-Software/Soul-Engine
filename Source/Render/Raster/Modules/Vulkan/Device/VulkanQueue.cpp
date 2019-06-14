#include "VulkanQueue.h"

VulkanQueue::VulkanQueue(const vk::Device& device, uint index, uint familyIndex):
	device_(device), index_(index), familyIndex_(familyIndex)
{

	queue_ = device_.getQueue(familyIndex_, index_);

}

bool VulkanQueue::Submit()
{

}

bool VulkanQueue::Present()
{

}

const vk::Queue& VulkanQueue::Get()
{

	return queue_;

}