#pragma once

#include "Display\Widget\Widget.h"
#include "Metrics.h"
#include "Raster Engine\RasterBackend.h"
#include "GPGPU\GPURasterBuffer.h"
#include "Engine Core\Camera\Camera.h"

class RenderWidget : public Widget
{
public:
	RenderWidget();
	~RenderWidget();

	virtual void Draw();
	virtual void RecreateData();

protected:



private:

	glm::uvec2 currentSize;
	GPURasterBuffer* buffer;

	uint samples;
	Camera* camera;
};

