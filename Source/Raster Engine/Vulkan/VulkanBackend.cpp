#include "VulkanBackend.h"
#include "Utility\Logger.h"
#include "VulkanUtility.h"

#include <vector>
#include <set>
#include <map>

/* The validation layers */
const std::vector<const char*> validationLayers = {
	"VK_LAYER_LUNARG_standard_validation"
};

/* The device extensions */
const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
/* True to enable, false to disable the validation layers */
const bool enableValidationLayers = false;
#else
/* True to enable, false to disable the validation layers */
const bool enableValidationLayers = true;
#endif


/* A queue family indices. */
struct QueueFamilyIndices {
	/* The graphics family */
	int graphicsFamily = -1;
	/* The present family */
	int presentFamily = -1;

	bool isComplete() {
		return graphicsFamily >= 0 && presentFamily >= 0;
	}
};

/* A swap chain support details. */
struct SwapChainSupportDetails {
	/* The capabilities */
	VkSurfaceCapabilitiesKHR capabilities;
	/* The formats */
	std::vector<VkSurfaceFormatKHR> formats;
	/* The present modes */
	std::vector<VkPresentModeKHR> presentModes;
};

/*
 *    ///////////////////////////Vulkan
 *    variables///////////////////////////////////////////////.
 *    @return	The instance.
 */

VulkanWrapper<VkInstance> instance{ vkDestroyInstance };

/*
 *    Gets the callback.
 *    @return	The callback.
 */

VulkanWrapper<VkDebugReportCallbackEXT> callback{ instance, DestroyDebugReportCallbackEXT };

/* Information about the vk window. */
struct VKWindowInformation {

	/*
	 *    Gets the surface.
	 *    @return	The surface.
	 */

	VulkanWrapper<VkSurfaceKHR> surface{ instance, vkDestroySurfaceKHR };
	/* The physical device */
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

	/*
	 *    Gets the device.
	 *    @return	The device.
	 */

	VulkanWrapper<VkDevice> device{ vkDestroyDevice };
};

/* The window storage */
static std::map<GLFWwindow*, std::unique_ptr< VKWindowInformation> > windowStorage;

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

/* ///////////////////////////////////////////////////////////////////////////////////. */
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

/* Sets up the debug callback. */
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

/*
 *    Creates a surface.
 *    @param [in,out]	window 	If non-null, the window.
 *    @param [in,out]	surface	If non-null, the surface.
 */

void CreateSurface(GLFWwindow* window, VkSurfaceKHR* surface) {
	if (glfwCreateWindowSurface(instance, window, nullptr, surface) != VK_SUCCESS) {
		S_LOG(S_ERROR, "Failed to create window surface");
	}
}

/*
 *    Device suitable.
 *    @param	device 	The device.
 *    @param	surface	The surface.
 *    @return	True if it succeeds, false if it fails.
 */

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

/*
 *    Swap chain support.
 *    @param	device 	The device.
 *    @param	surface	The surface.
 *    @return	The SwapChainSupportDetails.
 */

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

/*
 *    Device extension support.
 *    @param	device	The device.
 *    @return	True if it succeeds, false if it fails.
 */

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

/*
 *    Physical device.
 *    @param	physicalDevice	The physical device.
 *    @param	surface		  	The surface.
 */

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

/*
 *    Gets required extensions.
 *    @return	Null if it fails, else the required extensions.
 */

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

/*
 *    Determines if we can check validation layer support.
 *    @return	True if it succeeds, false if it fails.
 */

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

/*
 *    Searches for the first queue families.
 *    @param	device 	The device.
 *    @param	surface	The surface.
 *    @return	The found queue families.
 */

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

/* Default constructor. */
VulkanBackend::VulkanBackend() {
	CreateInstance();
	SetupDebugCallback();
}

/* Destructor. */
VulkanBackend::~VulkanBackend() {

}

/*
 *    Resize window.
 *    @param [in,out]	win	If non-null, the window.
 *    @param 		 	x  	The x coordinate.
 *    @param 		 	y  	The y coordinate.
 */

void VulkanBackend::ResizeWindow(GLFWwindow* win, int x, int y) {

}

/*
 *    Builds a window.
 *    @param [in,out]	window	If non-null, the window.
 */

void VulkanBackend::BuildWindow(GLFWwindow* window) {
	windowStorage[window] = std::unique_ptr<VKWindowInformation>(new VKWindowInformation());
	CreateSurface(window, windowStorage[window]->surface.replace());
	PhysicalDevice(windowStorage[window]->physicalDevice, windowStorage[window]->surface);
}

/* Sets window hints. */
void VulkanBackend::SetWindowHints() {
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
}

/*
 *    Pre raster.
 *    @param [in,out]	window	If non-null, the window.
 */

void VulkanBackend::PreRaster(GLFWwindow* window) {

}

/*
 *    Posts a raster.
 *    @param [in,out]	window	If non-null, the window.
 */

void VulkanBackend::PostRaster(GLFWwindow* window) {

}

/*
 *    Draws.
 *    @param [in,out]	window	If non-null, the window.
 *    @param [in,out]	job   	If non-null, the job.
 */

void VulkanBackend::Draw(GLFWwindow* window, RasterJob* job) {

}

/*
 *    Gets resource context.
 *    @return	Null if it fails, else the resource context.
 */

GLFWwindow* VulkanBackend::GetResourceContext() {
	return nullptr;
}
