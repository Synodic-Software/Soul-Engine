#include "VulkanBackend.h"
#include "Utility\Logger.h"
#include "VulkanUtility.h"

#include <vector>
#include <set>

const std::vector<const char*> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif


struct QueueFamilyIndices {
	int graphicsFamily = -1;
	int presentFamily = -1;

	bool isComplete() {
		return graphicsFamily >= 0 && presentFamily >= 0;
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

//////////////////////////////Vulkan variables///////////////////////////////////////////////

VulkanWrapper<VkInstance> instance{ vkDestroyInstance };
VulkanWrapper<VkDebugReportCallbackEXT> callback{ instance, DestroyDebugReportCallbackEXT };

struct VKWindowInformation {
	VulkanWrapper<VkSurfaceKHR> surface{ instance, vkDestroySurfaceKHR };
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VulkanWrapper<VkDevice> device{ vkDestroyDevice };
};

/////////////////////////Function Declarations/////////////////////////////////////////

void CreateInstance();
void SetupDebugCallback();
void CreateSurface(GLFWwindow*);
void PhysicalDevice(VkPhysicalDevice&);

SwapChainSupportDetails SwapChainSupport(const VkPhysicalDevice device, VkSurfaceKHR surface);
bool DeviceExtensionSupport(const VkPhysicalDevice device);
QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);
bool CheckValidationLayerSupport();
std::vector<const char*> GetRequiredExtensions();

//////////////////////////////////////////////////////////////////////////////////////
void CreateInstance() {

	if (enableValidationLayers && !CheckValidationLayerSupport()) {
		S_LOG(S_ERROR, "Validation layers not available");
	}

	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Soul Engine";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "Soul Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_MAKE_VERSION(1, 0, 30);

	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;

	auto extensions = GetRequiredExtensions();
	createInfo.enabledExtensionCount = extensions.size();
	createInfo.ppEnabledExtensionNames = extensions.data();

	if (enableValidationLayers) {
		createInfo.enabledLayerCount = validationLayers.size();
		createInfo.ppEnabledLayerNames = validationLayers.data();
	}
	else {
		createInfo.enabledLayerCount = 0;
	}

	if (vkCreateInstance(&createInfo, nullptr, instance.replace()) != VK_SUCCESS) {
		S_LOG(S_FATAL, "Failed to create vulkan instance");
	}
}

void SetupDebugCallback() {
	if (!enableValidationLayers) return;

	VkDebugReportCallbackCreateInfoEXT createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
	createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
	createInfo.pfnCallback = DebugCallback;

	if (CreateDebugReportCallbackEXT(instance, &createInfo, nullptr, callback.replace()) != VK_SUCCESS) {
		S_LOG(S_ERROR, "Failed to set up debug callback");
	}
}

void CreateSurface(GLFWwindow* window, VkSurfaceKHR* surface) {
	if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS) {
		S_LOG(S_ERROR, "Failed to create window surface");
	}
}

bool DeviceSuitable(const VkPhysicalDevice device, VkSurfaceKHR surface) {
	QueueFamilyIndices indices = FindQueueFamilies(device, surface);

	bool extensionsSupported = DeviceExtensionSupport(device);

	bool swapChainAdequate = false;
	if (extensionsSupported) {
		SwapChainSupportDetails swapChainSupport = SwapChainSupport(device, surface);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

SwapChainSupportDetails SwapChainSupport(const VkPhysicalDevice device, VkSurfaceKHR surface) {
	SwapChainSupportDetails details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	if (formatCount != 0) {
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

	if (presentModeCount != 0) {
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

bool DeviceExtensionSupport(const VkPhysicalDevice device) {
	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

void PhysicalDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

	if (deviceCount == 0) {
		S_LOG(S_ERROR, "Failed to find GPUs with Vulkan support");
	}

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	for (const auto& device : devices) {
		if (DeviceSuitable(device, surface)) {
			physicalDevice = device;
			break;
		}
	}

	if (physicalDevice == VK_NULL_HANDLE) {
		S_LOG(S_ERROR, "Failed to find a suitable GPU");
	}
}

std::vector<const char*> GetRequiredExtensions() {
	std::vector<const char*> extensions;

	unsigned int glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	if (!glfwExtensions) {
		S_LOG(S_FATAL, "Could not grab GLFW Vulkan Extensions");
	}

	for (unsigned int i = 0; i < glfwExtensionCount; i++) {
		extensions.push_back(glfwExtensions[i]);
	}

	if (enableValidationLayers) {
		extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
	}

	return extensions;
}

bool CheckValidationLayerSupport() {
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (const char* layerName : validationLayers) {
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) {
			return false;
		}
	}

	return true;
}

QueueFamilyIndices FindQueueFamilies(const VkPhysicalDevice device, VkSurfaceKHR surface) {
	QueueFamilyIndices indices;

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	int i = 0;
	for (const auto& queueFamily : queueFamilies) {
		if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			indices.graphicsFamily = i;
		}

		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

		if (queueFamily.queueCount > 0 && presentSupport) {
			indices.presentFamily = i;
		}

		if (indices.isComplete()) {
			break;
		}

		i++;
	}

	return indices;
}

VulkanBackend::VulkanBackend() {

}

VulkanBackend::~VulkanBackend() {

}

void VulkanBackend::Init() {
	CreateInstance();
	SetupDebugCallback();
}

void VulkanBackend::ResizeWindow(GLFWwindow* win, int x, int y) {

}

void VulkanBackend::BuildWindow(GLFWwindow* window) {

	VKWindowInformation info{};
	CreateSurface(window, info.surface.replace());
	PhysicalDevice(info.physicalDevice, info.surface);
}

void VulkanBackend::SetWindowHints() {
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
}

void VulkanBackend::PreRaster(GLFWwindow* window) {

}

void VulkanBackend::PostRaster(GLFWwindow* window) {

}

void VulkanBackend::Terminate() {
	//VulkanBackend::GetInstance().IdleDevice();
}

void VulkanBackend::Draw(GLFWwindow* window) {

}