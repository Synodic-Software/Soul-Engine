#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "VulkanUtility.h"
#include "Utility\GLMIncludes.h"


VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
	auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pCallback);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
	if (func != nullptr) {
		func(instance, callback, pAllocator);
	}
}


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

struct VulkanVertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(VulkanVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(VulkanVertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(VulkanVertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(VulkanVertex, texCoord);

		return attributeDescriptions;
	}

	bool operator==(const VulkanVertex& other) const {
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

namespace std {
	template<> struct hash<VulkanVertex> {
		size_t operator()(VulkanVertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

class VulkanBackend{
private:
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














	std::vector<GLFWwindow*> windows;


	VulkanWrapper<VkInstance> instance{ vkDestroyInstance };
	VulkanWrapper<VkDebugReportCallbackEXT> callback{ instance, DestroyDebugReportCallbackEXT };
	VulkanWrapper<VkSurfaceKHR> surface{ instance, vkDestroySurfaceKHR };

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VulkanWrapper<VkDevice> device{ vkDestroyDevice };

	VkQueue graphicsQueue;
	VkQueue presentQueue;

	VulkanWrapper<VkSwapchainKHR> swapChain{ device, vkDestroySwapchainKHR };
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VulkanWrapper<VkImageView>> swapChainImageViews;
	std::vector<VulkanWrapper<VkFramebuffer>> swapChainFramebuffers;

	VulkanWrapper<VkRenderPass> renderPass{ device, vkDestroyRenderPass };
	VulkanWrapper<VkDescriptorSetLayout> descriptorSetLayout{ device, vkDestroyDescriptorSetLayout };
	VulkanWrapper<VkPipelineLayout> pipelineLayout{ device, vkDestroyPipelineLayout };
	VulkanWrapper<VkPipeline> graphicsPipeline{ device, vkDestroyPipeline };

	VulkanWrapper<VkCommandPool> commandPool{ device, vkDestroyCommandPool };

	VulkanWrapper<VkImage> depthImage{ device, vkDestroyImage };
	VulkanWrapper<VkDeviceMemory> depthImageMemory{ device, vkFreeMemory };
	VulkanWrapper<VkImageView> depthImageView{ device, vkDestroyImageView };

	VulkanWrapper<VkImage> textureImage{ device, vkDestroyImage };
	VulkanWrapper<VkDeviceMemory> textureImageMemory{ device, vkFreeMemory };
	VulkanWrapper<VkImageView> textureImageView{ device, vkDestroyImageView };
	VulkanWrapper<VkSampler> textureSampler{ device, vkDestroySampler };

	VulkanWrapper<VkBuffer> vertexBuffer{ device, vkDestroyBuffer };
	VulkanWrapper<VkDeviceMemory> vertexBufferMemory{ device, vkFreeMemory };
	VulkanWrapper<VkBuffer> indexBuffer{ device, vkDestroyBuffer };
	VulkanWrapper<VkDeviceMemory> indexBufferMemory{ device, vkFreeMemory };

	VulkanWrapper<VkBuffer> uniformStagingBuffer{ device, vkDestroyBuffer };
	VulkanWrapper<VkDeviceMemory> uniformStagingBufferMemory{ device, vkFreeMemory };
	VulkanWrapper<VkBuffer> uniformBuffer{ device, vkDestroyBuffer };
	VulkanWrapper<VkDeviceMemory> uniformBufferMemory{ device, vkFreeMemory };

	VulkanWrapper<VkDescriptorPool> descriptorPool{ device, vkDestroyDescriptorPool };
	VkDescriptorSet descriptorSet;

	std::vector<VkCommandBuffer> commandBuffers;

	VulkanWrapper<VkSemaphore> imageAvailableSemaphore{ device, vkDestroySemaphore };
	VulkanWrapper<VkSemaphore> renderFinishedSemaphore{ device, vkDestroySemaphore };


public:
	void OnWindowResized(GLFWwindow* window, int width, int height);

	void RecreateSwapChain();

	void CreateInstance();

	void SetupDebugCallback();

	void CreateSurface(GLFWwindow* window);

	void PickVulkanDevice();

	void CreateVulkanLogical();

	void CreateSwapChain();

	void CreateImageViews();

	void CreateRenderPass();

	void CreateDescriptorSetLayout();

	void CreateGraphicsPipeline();

	void CreateFramebuffers();

	void CreateCommandPool();

	void CreateDepthResources();

	VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	VkFormat FindDepthFormat();

	void CreateTextureImage();

	void CreateTextureImageView();

	void CreateTextureSampler();

	void CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VulkanWrapper<VkImageView>& imageView);

	void CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VulkanWrapper<VkImage>& image, VulkanWrapper<VkDeviceMemory>& imageMemory);

	void TransitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout);

	void CopyImage(VkImage srcImage, VkImage dstImage, uint32_t width, uint32_t height);

	void LoadModel();

	void CreateVertexBuffer();

	void CreateIndexBuffer();

	void CreateUniformBuffer();

	void CreateDescriptorPool();

	void CreateDescriptorSet();

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VulkanWrapper<VkBuffer>& buffer, VulkanWrapper<VkDeviceMemory>& bufferMemory);

	VkCommandBuffer beginSingleTimeCommands();

	void EndSingleTimeCommands(VkCommandBuffer commandBuffer);

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	void CreateCommandBuffers();

	void CreateSemaphores();

	void UpdateUniformBuffer();

	void DrawFrame();

	void CreateShaderModule(const std::vector<char>& code, VulkanWrapper<VkShaderModule>& shaderModule);

	VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

	VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes);

	VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

	SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device);

	bool IsDeviceSuitable(VkPhysicalDevice device);

	bool CheckDeviceExtensionSupport(VkPhysicalDevice device);

	QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device);

	std::vector<const char*> GetRequiredExtensions();
	bool CheckValidationLayerSupport();
	static std::vector<char> ReadFile(const std::string& filename);

	static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData);




};

