//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Display\Widget\Widget.cpp.
//Implements the widget class.

#include "Widget.h"
#include "Raster Engine\RasterBackend.h"

//Default constructor.
Widget::Widget()
{
}

//Destructor.
Widget::~Widget()
{

}

//Draws this object.
void Widget::Draw()
{

}

//---------------------------------------------------------------------------------------------------
//Updates the window described by winIn.
//@param [in,out]	winIn	If non-null, the window in.

void Widget::UpdateWindow(GLFWwindow* winIn) {
	window = winIn;
}

//---------------------------------------------------------------------------------------------------
//Updates the positioning.
//@param	newPosition	The new position.
//@param	newSize	   	Size of the new.

void Widget::UpdatePositioning( glm::uvec2 newPosition, glm::uvec2 newSize) {
	size = newSize;
	position = newPosition;
}

//Recreate data.
void Widget::RecreateData() {

}
