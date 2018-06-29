#pragma once

#include "Rasterer\RasterJob.h"

/* A widget. */
class Widget
{
public:

	Widget() = default;
	virtual ~Widget() = default;

	virtual void Draw();

	virtual void UpdatePositioning(glm::uvec2, glm::uvec2);

	virtual void RecreateData();

protected:

	RasterJob* widgetJob;

	glm::uvec2 size;
	glm::uvec2 position;

private:

};

