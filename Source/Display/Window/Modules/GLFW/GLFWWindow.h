#pragma once

#include "Display/Window/Window.h"
#include "Core/Utility/ID/TypeID.h"

#include <memory>

struct GLFWmonitor;
struct GLFWwindow;
class RasterModule;
class VulkanRasterBackend;


class GLFWWindow final : public Window, TypeID<GLFWWindow>
{

public:

	GLFWWindow(const WindowParameters&, GLFWmonitor*, std::shared_ptr<RasterModule>, bool);
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

	std::shared_ptr<VulkanRasterBackend> rasterModule_;
	GLFWwindow* context_;

	bool master_;


	Entity surface_;
	Entity swapChain_;

	//TODO: Refactor to remove implementation specific code
	//vk::SurfaceKHR surface_;
	//std::unique_ptr<VulkanSwapChain> swapChain_;

};
