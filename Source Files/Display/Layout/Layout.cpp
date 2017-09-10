#include "Layout.h"

/* Default constructor. */
Layout::Layout()
{
}

/* Destructor. */
Layout::~Layout()
{
}

/* Draws this object. */
void Layout::Draw() {
	for (auto& wid : widgets) {
		wid->Draw();
	}
}

/*
 *    Updates the window described by winIn.
 *    @param [in,out]	winIn	If non-null, the window in.
 */

void Layout::UpdateWindow(GLFWwindow* winIn) {
	window = winIn;

	for (auto& wid : widgets) {
		wid->UpdateWindow(winIn);
	}
}

/*
 *    Updates the positioning.
 *    @param	newPosition	The new position.
 *    @param	newSize	   	Size of the new.
 */

void Layout::UpdatePositioning( glm::uvec2 newPosition, glm::uvec2 newSize) {
	size = newSize;
	position = newPosition;
	for (auto& wid : widgets) {
		wid->UpdatePositioning(newPosition, newSize);
	}

}

/* Recreate data. */
void Layout::RecreateData() {
	for (auto& wid : widgets) {
		wid->RecreateData();
	}
}
