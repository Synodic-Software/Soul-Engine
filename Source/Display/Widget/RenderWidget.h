#pragma once

#include "Display\Widget\Widget.h"

#include "Metrics.h"
#include "Compute\ComputeRasterBuffer.h"
#include "Compute\ComputeBuffer.h"

/* A render widget. */
class RenderWidget : public Widget
{
public:

	RenderWidget(uint&);
	~RenderWidget() = default;

	void Draw() override;

	void RecreateData() override;


private:

	glm::uvec2 currentSize;
	glm::uvec2 renderSize;

	ComputeRasterBuffer<glm::vec4> buffer;
	ComputeBuffer<glm::vec4> accumulator;
	ComputeBuffer<int> extraData;

	float samples;

	uint rayJob;

	uint iCounter;
	bool integrate;
	double time;
};

