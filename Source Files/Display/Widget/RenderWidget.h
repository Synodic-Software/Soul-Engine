#pragma once

#include "Display\Widget\Widget.h"
#include "Metrics.h"
#include "Raster Engine\RasterBackend.h"
#include "GPGPU\GPURasterBuffer.h"
#include "GPGPU\GPUBuffer.h"
#include "Engine Core\Camera\Camera.h"
#include "Ray Engine/RayEngine.h"

/* A render widget. */
/* A render widget. */
class RenderWidget : public Widget
{
public:

	/*
	 *    Constructor.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	RenderWidget(Camera*);
	/* Destructor. */
	/* Destructor. */
	~RenderWidget();

	/* Draws this RenderWidget. */
	/* Draws this RenderWidget. */
	virtual void Draw();
	/* Recreate data. */
	/* Recreate data. */
	virtual void RecreateData();

protected:



private:

	/* Size of the current */
	/* Size of the current */
	glm::uvec2 currentSize;
	/* Size of the render */
	/* Size of the render */
	glm::uvec2 renderSize;

	/* The buffer */
	/* The buffer */
	GPURasterBuffer* buffer;
	/* The accumulator */
	/* The accumulator */
	GPUBuffer* accumulator;
	/* Information describing the extra */
	/* Information describing the extra */
	GPUBuffer* extraData;

	/* The samples */
	/* The samples */
	float samples;
	/* The camera */
	/* The camera */
	Camera* camera;
	/* The ray job */
	/* The ray job */
	RayJob* rayJob;

	/* Zero-based index of the counter */
	/* Zero-based index of the counter */
	uint iCounter;
	/* True to integrate */
	/* True to integrate */
	bool integrate;
	/* The time */
	/* The time */
	double time;
};

