#pragma once

#include "Rasterer\RasterJob.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

/* A widget. */
class Widget
{
public:

	Widget() = default;
	virtual ~Widget() = default;

	virtual void Draw();

	virtual void UpdateWindow(GLFWwindow*);

	virtual void UpdatePositioning(glm::uvec2, glm::uvec2);

	virtual void RecreateData();

protected:

	RasterJob* widgetJob;
	GLFWwindow* window;

	glm::uvec2 size;
	glm::uvec2 position;

private:

};

