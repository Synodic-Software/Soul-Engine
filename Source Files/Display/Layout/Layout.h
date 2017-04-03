#pragma once

#include "Display\Widget\Widget.h"

#include "Metrics.h"
#include <list>
#include <memory>

class Layout : public Widget
{

public:
	Layout();
	~Layout();

	virtual void Draw();
	virtual void UpdateWindow(GLFWwindow*);
	virtual void UpdatePositioning(glm::uvec2, glm::uvec2);
	virtual void RecreateData();

protected:

	std::list<Widget*> widgets;

private:
};

