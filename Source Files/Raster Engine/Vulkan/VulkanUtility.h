#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Utility\Includes\GLMIncludes.h"
#include "Utility\Logger.h"

#include <glm/gtx/hash.hpp>

#include <functional>
#include <array>
#include <iostream>
#include <stdexcept>

template <typename T>
class VulkanWrapper {
public:
	/* Default constructor. */
	/* Default constructor. */
	VulkanWrapper() : VulkanWrapper([](T, VkAllocationCallbacks*) {}) {}

	/*
	 *    Constructor.
	 *
	 *    @param [in,out]	deletef	If non-null, the deletef.
	 */

	VulkanWrapper(std::function<void(T, VkAllocationCallbacks*)> deletef) {
		this->deleter = [=](T obj) { deletef(obj, nullptr); };
	}

	/*
	 *    Constructor.
	 *
	 *    @param 		 	instance	The instance.
	 *    @param [in,out]	deletef 	If non-null, the deletef.
	 */

	VulkanWrapper(const VulkanWrapper<VkInstance>& instance, std::function<void(VkInstance, T, VkAllocationCallbacks*)> deletef) {
		this->deleter = [&instance, deletef](T obj) { deletef(instance, obj, nullptr); };
	}

	/*
	 *    Constructor.
	 *
	 *    @param 		 	device 	The device.
	 *    @param [in,out]	deletef	If non-null, the deletef.
	 */

	VulkanWrapper(const VulkanWrapper<VkDevice>& device, std::function<void(VkDevice, T, VkAllocationCallbacks*)> deletef) {
		this->deleter = [&device, deletef](T obj) { deletef(device, obj, nullptr); };
	}

	/* Destructor. */
	/* Destructor. */
	~VulkanWrapper() {
		/* Default constructor. */
		cleanup();
	}

	/*
	 *    Reference operator.
	 *
	 *    @return	The result of the operation.
	 */

	const T* operator &() const {
		return &object;
	}

	/*
	 *    Gets the replace.
	 *
	 *    @return	Null if it fails, else a pointer to a T.
	 */

	T* replace() {
		cleanup();
		return &object;
	}

	/*
	 *    Cast that converts the given  to a T.
	 *
	 *    @return	The result of the operation.
	 */

	operator T() const {
		return object;
	}

	/*
	 *    Assignment operator.
	 *
	 *    @param	rhs	The right hand side.
	 */

	void operator=(T rhs) {
		if (rhs != object) {
			cleanup();
			object = rhs;
		}
	}

	template<typename V>

	/*
	 *    Equality operator.
	 *
	 *    @param	rhs	The right hand side.
	 *
	 *    @return	True if the parameters are considered equivalent.
	 */

	bool operator==(V rhs) {
		return object == T(rhs);
	}

private:

	/*
	 *    Gets the object.
	 *
	 *    @return	The object.
	 */

	T object{ VK_NULL_HANDLE };
	/* The deleter */
	/* The deleter */
	std::function<void(T)> deleter;

	/* Cleanups this VulkanWrapper. */
	/* Cleanups this VulkanWrapper. */
	void cleanup() {
		if (object != VK_NULL_HANDLE) {
			deleter(object);
		}
		object = VK_NULL_HANDLE;
	}
};


/* A vertex. */
/* A vertex. */
struct Vertex {
	/* The position */
	/* The position */
	glm::vec3 pos;
	/* The color */
	/* The color */
	glm::vec3 color;
	/* The tex coordinate */
	/* The tex coordinate */
	glm::vec2 texCoord;

	/*
	 *    Gets binding description.
	 *
	 *    @return	The binding description.
	 */

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	/*
	 *    Gets attribute descriptions.
	 *
	 *    @return	The attribute descriptions.
	 */

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}

	/*
	 *    Equality operator.
	 *
	 *    @param	other	The other.
	 *
	 *    @return	True if the parameters are considered equivalent.
	 */

	bool operator==(const Vertex& other) const {
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}
};

/* . */
/* . */
namespace std {
	/* A hash. */
	/* A hash. */
	template<> struct hash<Vertex> {

		/*
		 *    Function call operator.
		 *
		 *    @param	vertex	The vertex.
		 *
		 *    @return	The result of the operation.
		 */

		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

/* An uniform buffer object. */
/* An uniform buffer object. */
struct UniformBufferObject {
	/* The model */
	/* The model */
	glm::mat4 model;
	/* The view */
	/* The view */
	glm::mat4 view;
	/* The project */
	/* The project */
	glm::mat4 proj;
};

/*
 *    Callback, called when the debug.
 *
 *    @param 		 	flags	   	The flags.
 *    @param 		 	objType	   	Type of the object.
 *    @param 		 	obj		   	The object.
 *    @param 		 	location   	The location.
 *    @param 		 	code	   	The code.
 *    @param 		 	layerPrefix	The layer prefix.
 *    @param 		 	msg		   	The message.
 *    @param [in,out]	userData   	If non-null, information describing the user.
 *
 *    @return	A VKAPI_CALL.
 */

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objType, uint64_t obj, size_t location, int32_t code, const char* layerPrefix, const char* msg, void* userData) {
	std::cerr << "Validation layer: " << msg << std::endl;
	S_LOG_ERROR("Validation layer: " , msg );
	return VK_FALSE;
}

/*
 *    Creates debug report callback extent.
 *
 *    @param 		 	instance   	The instance.
 *    @param 		 	pCreateInfo	Information describing the create.
 *    @param 		 	pAllocator 	The allocator.
 *    @param [in,out]	pCallback  	If non-null, the callback.
 *
 *    @return	The new debug report callback extent.
 */

VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
	auto func = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pCallback);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

/*
 *    Destroys the debug report callback extent.
 *
 *    @param	instance  	The instance.
 *    @param	callback  	The callback.
 *    @param	pAllocator	The allocator.
 */

void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
	if (func != nullptr) {
		func(instance, callback, pAllocator);
	}
}