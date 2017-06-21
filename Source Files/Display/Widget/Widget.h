#pragma once

#include "Metrics.h"
#include "Raster Engine\RasterJob.h"
#include "Utility\Includes\GLFWIncludes.h"

/* A widget. */
class Widget
{
public:
	/* Default constructor. */
	Widget();
	/* Destructor. */
	~Widget();

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
	

	/* The widget job */
	RasterJob* widgetJob;
	/* The window */
	GLFWwindow* window;

	/* The size */
	glm::uvec2 size;
	/* The position */
	glm::uvec2 position;

private:

};

