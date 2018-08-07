#include "VulkanSurface.h"

#include "VulkanContext.h"

#include "GLFW/glfw3.h"

VulkanSurface::VulkanSurface(VulkanContext* context, std::any& windowContext):
	context_(context)
{

	//GLFW uses vulkan.h 'c version'
	VkSurfaceKHR castSurface;

	//garunteed to use GLFW if using vulkan (TODO: cross platform alternative)
	const VkResult error = glfwCreateWindowSurface(
		static_cast<VkInstance>(context_->GetInstance()),
		std::any_cast<GLFWwindow*>(windowContext), //TODO: abstract the context
		nullptr,
		&castSurface
	);

	//back to c++ land
	surface_ = static_cast<vk::SurfaceKHR>(castSurface);

	assert(error == VK_SUCCESS);

}

void VulkanSurface::Terminate() {

	const vk::Instance& instance = context_->GetInstance();
	instance.destroySurfaceKHR(surface_);

}

vk::SurfaceKHR& VulkanSurface::GetSurface() {
	return surface_;
}
