#pragma once

#include "Display/Window.h"
#include "Core/Utility/ID/TypeID.h"

struct GLFWmonitor;
struct GLFWwindow;
class VulkanRasterBackend;

class GLFWWindow final : public Window, TypeID<GLFWWindow>
{

public:

	GLFWWindow(const WindowParameters&, GLFWmonitor*, VulkanRasterBackend*, bool);
	~GLFWWindow() override;

	GLFWWindow(const GLFWWindow &) = delete;
	GLFWWindow(GLFWWindow &&) noexcept = default;

	GLFWWindow& operator=(const GLFWWindow &) = delete;
	GLFWWindow& operator=(GLFWWindow &&) noexcept = default;


	GLFWwindow* Context() const;
	bool Master() const;

private:
	VulkanRasterBackend* rasterModule_;
	GLFWwindow* context_;

	bool master_;

};
