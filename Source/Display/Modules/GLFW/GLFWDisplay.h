#pragma once

#include "Display/DisplayModule.h"

#include <vector>
#include <unordered_map>

struct GLFWmonitor;
struct GLFWwindow;
class GLFWWindow;
class WindowParameters;
class VulkanRasterBackend;

class GLFWDisplay final : public DisplayModule {

public:

	GLFWDisplay();
	~GLFWDisplay() override;

	GLFWDisplay(const GLFWDisplay&) = delete;
	GLFWDisplay(GLFWDisplay&&) noexcept = default;

	GLFWDisplay& operator=(const GLFWDisplay&) = delete;
	GLFWDisplay& operator=(GLFWDisplay&&) noexcept = default;

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
