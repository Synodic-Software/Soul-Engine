#include "SingleLayout.h"
//#include "Raster Engine\RasterManager.h"

/*
 *    Constructor.
 *    @param [in,out]	widget	If non-null, the widget.
 */

SingleLayout::SingleLayout(Widget* widget)
{
	widgets.push_back(widget);
}

/* Destructor. */
SingleLayout::~SingleLayout()
{

}

/* Draws this object. */
void SingleLayout::Draw() {
	Layout::Draw();
}