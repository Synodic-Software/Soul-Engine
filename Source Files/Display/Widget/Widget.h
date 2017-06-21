#pragma once

#include "Metrics.h"
#include "Raster Engine\RasterJob.h"
#include "Utility\Includes\GLFWIncludes.h"

/* A widget. */
/* A widget. */
class Widget
{
public:
	/* Default constructor. */
	/* Default constructor. */
	Widget();
	/* Destructor. */
	/* Destructor. */
	~Widget();

	/* Draws this Widget. */
	/* Draws this Widget. */
	virtual void Draw();

	/*
	 *    Updates the window described by parameter1.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	virtual void UpdateWindow(GLFWwindow*);

	/*
	 *    Updates the positioning.
	 *
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 */

	virtual void UpdatePositioning(glm::uvec2, glm::uvec2);
	/* Recreate data. */
	/* Recreate data. */
	virtual void RecreateData();

protected:
	

	/* The widget job */
	/* The widget job */
	RasterJob* widgetJob;
	/* The window */
	/* The window */
	GLFWwindow* window;

	/* The size */
	/* The size */
	glm::uvec2 size;
	/* The position */
	/* The position */
	glm::uvec2 position;

private:

};

