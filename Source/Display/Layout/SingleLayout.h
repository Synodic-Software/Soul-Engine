#pragma once

#include "Display\Layout\Layout.h"
#include "Metrics.h"

/* A single layout. */
class SingleLayout : public Layout
{
public:

	/*
	 *    Constructor.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	SingleLayout(Widget*);
	/* Destructor. */
	~SingleLayout();

	/* Draws this object. */
	virtual void Draw();
protected:



private:

};

