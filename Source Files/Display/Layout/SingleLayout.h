#pragma once

#include "Display\Layout\Layout.h"
#include "Metrics.h"

class SingleLayout : public Layout
{
public:
	SingleLayout(Widget*);
	SingleLayout(GLFWwindow*, glm::uvec2,Widget*);
	~SingleLayout();

	virtual void Draw(GLFWwindow*);
protected:



private:

};

