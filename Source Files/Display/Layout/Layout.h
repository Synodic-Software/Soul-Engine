#pragma once

#include "Display\Widget\Widget.h"

#include "Metrics.h"
#include <list>
#include <memory>

class Layout : public Widget
{

public:
	Layout();
	Layout(GLFWwindow*, glm::uvec2);
	~Layout();

	virtual void Draw(GLFWwindow*);
	virtual void UpdateWindow(GLFWwindow*);

protected:

	std::list<Widget*> widgets;

private:
};

