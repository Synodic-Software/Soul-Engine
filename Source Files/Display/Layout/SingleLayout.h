#pragma once

#include "Display\Layout\Layout.h"
#include "Metrics.h"

class SingleLayout : public Layout
{
public:
	SingleLayout(Widget*);
	~SingleLayout();

	virtual void Draw();
protected:



private:

};

