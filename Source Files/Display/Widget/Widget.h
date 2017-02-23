#pragma once

#include "Metrics.h"
#include "Raster Engine\RasterJob.h"
#include "Raster Engine\RasterBackend.h"

class Widget
{
public:
	Widget();
	~Widget();

	virtual void Draw()=0;

protected:
	RasterJob* widgetJob;

private:

};

