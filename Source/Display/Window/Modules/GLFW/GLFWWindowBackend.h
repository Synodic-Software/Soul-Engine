#pragma once

#include "Display/Window/WindowModule.h"

#include <unordered_map>

struct GLFWmonitor;
struct GLFWwindow;
class GLFWWindow;
class WindowParameters;
class VulkanRasterBackend;
class GLFWInputBackend;

class GLFWWindowBackend final : public WindowModule {

public:

	GLFWWindowBackend(std::shared_ptr<InputModule>&);
	~GLFWWindowBackend() override;

	GLFWWindowBackend(const GLFWWindowBackend&) = delete;
	GLFWWindowBackend(GLFWWindowBackend&&) noexcept = default;

	GLFWWindowBackend& operator=(const GLFWWindowBackend&) = delete;
	GLFWWindowBackend& operator=(GLFWWindowBackend&&) noexcept = default;

	void Update() override;
	void Draw() override;
	bool Active() override;
	void CreateWindow(const WindowParameters&, std::shared_ptr<RasterModule>&) override;

	std::vector<const char*> GetRasterExtensions() override;

	Window& GetWindow() override;


	void Refresh();
	void Resize(int, int);
	void PositionUpdate(int, int);
	void FrameBufferResize(GLFWWindow&, int, int);
	void Close(GLFWWindow&);

	struct UserPointers {
		GLFWWindowBackend* windowBackend;
		std::shared_ptr<GLFWInputBackend> inputBackend;
	};

private:

	GLFWWindow& GetWindow(GLFWwindow*);

	//TODO: Replace with std::span as GLFW owns and manages the monitors - C++20
	std::vector<GLFWmonitor*> monitors_;
	std::unordered_map<GLFWwindow*, std::unique_ptr<GLFWWindow>> windows_;

	// Luxery of having the Input system be GLFW
	std::shared_ptr<GLFWInputBackend> inputModule_;

	UserPointers userPointers_;


};
