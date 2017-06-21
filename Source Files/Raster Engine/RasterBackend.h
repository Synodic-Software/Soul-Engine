#pragma once

#include "Display\Window\Window.h"
#include "Raster Engine\Shader.h"
#include "Raster Engine\RasterJob.h"
#include "Raster Engine\Buffer.h"

#include "RasterBase.h"
#include "OpenGL\OpenGLBackend.h"
#include "Vulkan\VulkanBackend.h"

/* Values that represent backend names. */
/* Values that represent backend names. */
enum BackendName { OpenGL, Vulkan };

/* . */
/* . */
namespace RasterBackend {

/* . */
/* . */
	namespace detail {
		/* The raster */
		/* The raster */
		extern std::unique_ptr<RasterBase> raster;
	}


	/////////////////////////////////////////
	/*         Public Definitions         */
	/* //////////////////////////////////////. */

	/* The backend */
	extern BackendName backend;


	/* Initializes this object. */
	/* Initializes this object. */
	void Initialize();

	/* needs to be called from the main thread. */
	/* Sets window hints. */
	void SetWindowHints();

	/*
	 *    Creates a shader.
	 *
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 *
	 *    @return	Null if it fails, else the new shader.
	 */

	Shader* CreateShader(const std::string&, shader_t);

	/*
	 *    Creates a buffer.
	 *
	 *    @param	size	The size.
	 *
	 *    @return	Null if it fails, else the new buffer.
	 */

	Buffer* CreateBuffer(uint size);

	/*
	 *    Creates the job.
	 *
	 *    @return	Null if it fails, else the new job.
	 */

	RasterJob* CreateJob();

	/* Makes context current. */
	/* Makes context current. */
	void MakeContextCurrent();

	/*
	 *    Resize window.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 *    @param 		 	parameter3	The third parameter.
	 */

	void ResizeWindow(GLFWwindow*, int, int);

	/*
	 *    Builds a window.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void BuildWindow(GLFWwindow*);

	/* Terminates this object. */
	/* Terminates this object. */
	void Terminate();

	/*
	 *    Draws.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param [in,out]	parameter2	If non-null, the second parameter.
	 */

	void Draw(GLFWwindow*, RasterJob*);

	/*
	 *    Pre raster.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void PreRaster(GLFWwindow*);

	/*
	 *    Posts a raster.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void PostRaster(GLFWwindow*);

}
