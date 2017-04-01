#pragma once

#include "Metrics.h"
#include "Raster Engine\RasterJob.h"
#include "Utility\Includes\GLFWIncludes.h"

class Widget
{
public:
	Widget();
	~Widget();

	virtual void Draw(GLFWwindow*);
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

