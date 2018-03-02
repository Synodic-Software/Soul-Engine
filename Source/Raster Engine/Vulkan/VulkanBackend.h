#pragma once

#include "Display\Window\Window.h"
#include "Raster Engine\RasterBase.h"


/* A vulkan backend. */
class VulkanBackend : public RasterBase {
public:
	/* Default constructor. */
	VulkanBackend();
	/* Destructor. */
	~VulkanBackend();

	/*
	 *    Gets resource context.
	 *    @return	Null if it fails, else the resource context.
	 */

	virtual GLFWwindow* GetResourceContext();
	/* Sets window hints. */
	virtual void SetWindowHints();

	/*
	 *    Resize window.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 *    @param 		 	parameter3	The third parameter.
	 */

	virtual void ResizeWindow(GLFWwindow*, int, int);

	/*
	 *    Builds a window.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	virtual void BuildWindow(GLFWwindow*);

	/*
	 *    Pre raster.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void PreRaster(GLFWwindow*);

	/*
	 *    Posts a raster.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void PostRaster(GLFWwindow*);

	/*
	 *    Draws.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param [in,out]	parameter2	If non-null, the second parameter.
	 */

	virtual void Draw(GLFWwindow*, RasterJob*);

	/*
	 *    Raster function.
	 *    @tparam	Fn  	Type of the function.
	 *    @tparam	Args	Type of the arguments.
	 *    @param [in,out]	fn  	The function.
	 *    @param 		 	args	Variable arguments providing [in,out] The arguments.
	 */

	template<typename Fn,
		typename ... Args>
		void RasterFunction(Fn && fn, Args && ... args) {


	}

private:

};