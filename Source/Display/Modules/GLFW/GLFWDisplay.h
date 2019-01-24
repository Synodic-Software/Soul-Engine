#pragma once

#include "Display/Display.h"

#include <vector>
#include <unordered_map>

struct GLFWmonitor;
struct GLFWwindow;
class GLFWWindow;
class WindowParameters;
class VulkanRasterBackend;

class GLFWDisplay final : public Display {

public:

	GLFWDisplay();
	~GLFWDisplay() override;

	GLFWDisplay(const GLFWDisplay&) = delete;
	GLFWDisplay(GLFWDisplay&&) noexcept = default;

	GLFWDisplay& operator=(const GLFWDisplay&) = delete;
	GLFWDisplay& operator=(GLFWDisplay&&) noexcept = default;

	void Draw() override;
	bool Active() override;
	void CreateWindow(const WindowParameters&, RasterBackend*) override;
	void RegisterRasterBackend(RasterBackend*) override;


	void Refresh();
	void Resize(int, int);
	void PositionUpdate(int, int);
	void FrameBufferResize(int, int);
	void Close(GLFWWindow&);


private:

	GLFWWindow& GetWindow(GLFWwindow*);

	//TODO: Replace with std::span as GLFW owns and manages the monitors - C++20
	std::vector<GLFWmonitor*> monitors_;
	std::unordered_map<GLFWwindow*, std::unique_ptr<GLFWWindow>> windows_;

};
