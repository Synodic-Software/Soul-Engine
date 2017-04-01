#pragma once

#include "Display\Widget\Widget.h"
#include "Metrics.h"

class RenderWidget : public Widget
{
public:
	RenderWidget();
	~RenderWidget();

	virtual void Draw();
	virtual void RecreateData();

protected:



private:

};

