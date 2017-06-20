//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Raster Engine\RasterBase.h.
//Declares the raster base class.

#pragma once

#include "Display\Window\Window.h"
#include "Raster Engine\Shader.h"
#include "Raster Engine\RasterJob.h"
#include "Raster Engine\Buffer.h"


//A raster base.
class RasterBase {
public:
	//Default constructor.
	RasterBase();
	//Destructor.
	~RasterBase();

	//Sets window hints.
	virtual void SetWindowHints() = 0;

	//---------------------------------------------------------------------------------------------------
	//Gets resource context.
	//@return	Null if it fails, else the resource context.

	virtual GLFWwindow* GetResourceContext() = 0;

	//---------------------------------------------------------------------------------------------------
	//Builds a window.
	//@param [in,out]	parameter1	If non-null, the first parameter.

	virtual void BuildWindow(GLFWwindow*) = 0;

	//---------------------------------------------------------------------------------------------------
	//Draws.
	//@param [in,out]	parameter1	If non-null, the first parameter.
	//@param [in,out]	parameter2	If non-null, the second parameter.

	virtual void Draw(GLFWwindow*, RasterJob*) = 0;

	//---------------------------------------------------------------------------------------------------
	//Resize window.
	//@param [in,out]	parameter1	If non-null, the first parameter.
	//@param 		 	parameter2	The second parameter.
	//@param 		 	parameter3	The third parameter.

	virtual void ResizeWindow(GLFWwindow*, int, int) = 0;

	//---------------------------------------------------------------------------------------------------
	//Pre raster.
	//@param [in,out]	parameter1	If non-null, the first parameter.

	virtual void PreRaster(GLFWwindow*) = 0;

	//---------------------------------------------------------------------------------------------------
	//Posts a raster.
	//@param [in,out]	parameter1	If non-null, the first parameter.

	virtual void PostRaster(GLFWwindow*) = 0;
	//Makes context current.
	virtual void MakeContextCurrent() = 0;

private:
};

