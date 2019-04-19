#pragma once

#include "Display/Window/WindowModule.h"

#include <vector>
#include <unordered_map>

struct GLFWmonitor;
struct GLFWwindow;
class GLFWWindow;
class WindowParameters;
class VulkanRasterBackend;

class GLFWWindowBackend final : public WindowModule {

public:

	GLFWWindowBackend();
	~GLFWWindowBackend() override;

	GLFWWindowBackend(const GLFWWindowBackend&) = delete;
	GLFWWindowBackend(GLFWWindowBackend&&) noexcept = default;

	GLFWWindowBackend& operator=(const GLFWWindowBackend&) = delete;
	GLFWWindowBackend& operator=(GLFWWindowBackend&&) noexcept = default;

	void Draw() override;
	bool Active() override;
	void CreateWindow(const WindowParameters&, RasterModule*) override;
	void RegisterRasterBackend(RasterModule*) override;


	void Refresh();
	void Resize(int, int);
	void PositionUpdate(int, int);
	void FrameBufferResize(GLFWWindow&, int, int);
	void Close(GLFWWindow&);


private:

	GLFWWindow& GetWindow(GLFWwindow*);

	//TODO: Replace with std::span as GLFW owns and manages the monitors - C++20
	std::vector<GLFWmonitor*> monitors_;
	std::unordered_map<GLFWwindow*, std::unique_ptr<GLFWWindow>> windows_;

};
