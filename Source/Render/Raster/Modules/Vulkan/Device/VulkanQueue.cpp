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

bool VulkanQueue::Present(
	nonstd::span<vk::Semaphore> semaphores,
	nonstd::span<vk::SwapchainKHR> swapchains,
	nonstd::span<uint> imageIndices
)
{

	assert(swapchains.size() == imageIndices.size());

	//TODO: Reduce allocation calls
	std::vector<vk::Result> swapchainResults(swapchains.size());

	vk::PresentInfoKHR presentInfo;
	presentInfo.waitSemaphoreCount = semaphores.size();
	presentInfo.pWaitSemaphores = semaphores.data();
	presentInfo.swapchainCount = swapchains.size();
	presentInfo.pSwapchains = swapchains.data();
	presentInfo.pImageIndices = imageIndices.data();
	presentInfo.pResults = swapchainResults.data();

	auto result = queue_.presentKHR(presentInfo);

	bool success = result == vk::Result::eSuccess;

	for (const auto& swapchainResult : swapchainResults) {

		success &= swapchainResult == vk::Result::eSuccess;

	}

	return success;

}

const vk::Queue& VulkanQueue::Handle()
{

	return queue_;
}

uint VulkanQueue::FamilyIndex() const
{

	return familyIndex_;
}