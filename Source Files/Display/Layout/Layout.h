#pragma once

#include "Display\Widget\Widget.h"

#include "Metrics.h"
#include <list>
#include <memory>

/* A layout. */
class Layout : public Widget
{

public:
	/* Default constructor. */
	Layout();
	/* Destructor. */
	~Layout();

	/* Draws this object. */
	virtual void Draw();

	/*
	 *    Updates the window described by parameter1.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	virtual void UpdateWindow(GLFWwindow*);

	/*
	 *    Updates the positioning.
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 */

	virtual void UpdatePositioning(glm::uvec2, glm::uvec2);
	/* Recreate data. */
	virtual void RecreateData();

protected:

	/* The widgets */
	std::list<Widget*> widgets;

private:
};

