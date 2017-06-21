#include "SingleLayout.h"
#include "Raster Engine\RasterBackend.h"

/*
 *    Constructor.
 *
 *    @param [in,out]	widget	If non-null, the widget.
 */

SingleLayout::SingleLayout(Widget* widget)
{
	widgets.push_back(widget);
}

/* Destructor. */
/* Destructor. */
SingleLayout::~SingleLayout()
{

}

/* Draws this SingleLayout. */
/* Draws this SingleLayout. */
void SingleLayout::Draw() {
	Layout::Draw();
}