#include "SingleLayout.h"
#include "Raster Engine\RasterBackend.h"

SingleLayout::SingleLayout(Widget* widget)
{
	widgets.push_back(widget);
}
SingleLayout::SingleLayout(GLFWwindow* winIn, Widget* widget)
: Layout(winIn){
	widgets.push_back(widget);
}

SingleLayout::~SingleLayout()
{

}

void SingleLayout::Draw(GLFWwindow* window) {
	Layout::Draw(window);
}