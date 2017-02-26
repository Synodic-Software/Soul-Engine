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

protected:
	

	RasterJob* widgetJob;
	GLFWwindow* window;

private:

};

