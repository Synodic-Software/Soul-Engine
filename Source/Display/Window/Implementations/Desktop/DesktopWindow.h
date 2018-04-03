#pragma once

#include "Display\Window\AbstractWindow.h"

class DesktopWindow : public AbstractWindow 
{
public:
	/* Constructor. */
	DesktopWindow(WindowType, const std::string&, uint x, uint y, uint width, uint height, void*, void*);

	/* Destructor. */
	~DesktopWindow();

	/* Draws this object. */
	void Draw();

};