#pragma once

#include "Display/Window.h"
#include "Core/Utility/ID/TypeID.h"
#include <vulkan/vulkan.hpp>

struct GLFWmonitor;
struct GLFWwindow;
class VulkanRasterBackend;
class VulkanSwapChain;

class GLFWWindow final : public Window, TypeID<GLFWWindow>
{

public:

	GLFWWindow(const WindowParameters&, GLFWmonitor*, VulkanRasterBackend*, bool);
	~GLFWWindow() override;

	GLFWWindow(const GLFWWindow &) = delete;
	GLFWWindow(GLFWWindow &&) noexcept = default;

	GLFWWindow& operator=(const GLFWWindow &) = delete;
	GLFWWindow& operator=(GLFWWindow &&) noexcept = default;


	void Draw();
	void FrameBufferResize(int, int);


	GLFWwindow* Context() const;
	bool Master() const;


private:

	VulkanRasterBackend* rasterModule_;
	GLFWwindow* context_;

	bool master_;


	vk::SurfaceKHR surface_;
	std::unique_ptr<VulkanSwapChain> swapChain_;

};
