#include "VulkanQueue.h"

VulkanQueue::VulkanQueue(const vk::Device& device, uint familyIndex, uint index):
	device_(device), familyIndex_(familyIndex), index_(index)
{

	queue_ = device_.getQueue(familyIndex_, index_);

}

bool VulkanQueue::Submit()
{

	return false;

}

bool VulkanQueue::Present()
{

	return false;

}

const vk::Queue& VulkanQueue::Handle()
{

	return queue_;

}

const uint VulkanQueue::FamilyIndex()
{

	return familyIndex_;

}