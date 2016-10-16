#pragma once

#include <vulkan/vulkan.h>

#include <stdexcept>
#include <functional>

template <typename T>
class VulkanWrapper {
public:
	VulkanWrapper() : VulkanWrapper([](T, VkAllocationCallbacks*) {}) {}

	VulkanWrapper(std::function<void(T, VkAllocationCallbacks*)> deletef) {
		this->deleter = [=](T obj) { deletef(obj, nullptr); };
	}

	VulkanWrapper(const VulkanWrapper<VkInstance>& instance, std::function<void(VkInstance, T, VkAllocationCallbacks*)> deletef) {
		this->deleter = [&instance, deletef](T obj) { deletef(instance, obj, nullptr); };
	}

	VulkanWrapper(const VulkanWrapper<VkDevice>& device, std::function<void(VkDevice, T, VkAllocationCallbacks*)> deletef) {
		this->deleter = [&device, deletef](T obj) { deletef(device, obj, nullptr); };
	}

	~VulkanWrapper() {
		cleanup();
	}

	const T* operator &() const {
		return &object;
	}

	T* replace() {
		cleanup();
		return &object;
	}

	operator T() const {
		return object;
	}

	void operator=(T rhs) {
		if (rhs != object) {
			cleanup();
			object = rhs;
		}
	}

	template<typename V>
	bool operator==(V rhs) {
		return object == T(rhs);
	}

private:
	T object{ VK_NULL_HANDLE };
	std::function<void(T)> deleter;

	void cleanup() {
		if (object != VK_NULL_HANDLE) {
			deleter(object);
		}
		object = VK_NULL_HANDLE;
	}
};