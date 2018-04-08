#include "Window.h"
#include "Utility/Logger.h"
#include "Raster Engine/RasterManager.h"
#include "Parallelism/Scheduler.h"
#include "WindowManager.h"
#include "Input/InputManager.h"

/*
 *    Constructor.
 *    @param 		 	inWin		 	The in window.
 *    @param 		 	inTitle		 	The in title.
 *    @param 		 	x			 	An uint to process.
 *    @param 		 	y			 	An uint to process.
 *    @param 		 	iwidth		 	The iwidth.
 *    @param 		 	iheight		 	The iheight.
 *    @param [in,out]	monitorIn	 	If non-null, the monitor in.
 *    @param [in,out]	sharedContext	If non-null, context for the shared.
 */

Window::Window(const std::string& inTitle, uint x, uint y, uint iwidth, uint iheight)
{
	xPos = x;
	yPos = y;
	width = iwidth;
	height = iheight;
	title = inTitle;

}