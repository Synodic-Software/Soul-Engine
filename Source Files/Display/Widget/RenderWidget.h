#pragma once

#include "Display\Widget\Widget.h"
#include "Metrics.h"
#include "Raster Engine\RasterBackend.h"
#include "GPGPU\GPURasterBuffer.h"
#include "GPGPU\GPUBuffer.h"
#include "Engine Core\Camera\Camera.h"
#include "Ray Engine/RayEngine.h"

class RenderWidget : public Widget
{
public:
	RenderWidget(Camera*);
	~RenderWidget();

	virtual void Draw();
	virtual void RecreateData();

protected:



private:

	void EarlyFrameUpdate();
	void LateFrameUpdate();

	glm::uvec2 currentSize;
	glm::uvec2 renderSize;

	GPURasterBuffer* buffer;
	GPUBuffer* accumulator;
	GPUBuffer* extraData;

	float samples;
	Camera* camera;
	RayJob* rayJob;

	uint iCounter;
	bool integrate;
	double time;
};

