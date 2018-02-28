#pragma once

#include "Display\Widget\Widget.h"
#include <list>

/* A layout. */
class Layout : public Widget
{

public:

	Layout() = default;
	virtual ~Layout() = default;


	void Draw() override;

	void UpdateWindow(GLFWwindow*) override;

	void UpdatePositioning(glm::uvec2, glm::uvec2) override;

	void RecreateData() override;

protected:


	std::list<Widget*> widgets;

private:
};

