#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBase.h"
#include "Utility\Includes\GLMIncludes.h"

#include <map>
#include <memory>

/*
 *    Glew get context.
 *
 *    @return	Null if it fails, else a pointer to a GLEWContext.
 */

GLEWContext* glewGetContext();

/* An open gl backend. */
/* An open gl backend. */
class OpenGLBackend : public RasterBase {
public:
	/* Default constructor. */
	/* Default constructor. */
	OpenGLBackend();
	/* Destructor. */
	/* Destructor. */
	~OpenGLBackend();

	/*
	 *    Gets resource context.
	 *
	 *    @return	Null if it fails, else the resource context.
	 */

	virtual GLFWwindow* GetResourceContext();
	/* Sets window hints. */
	/* Sets window hints. */
	virtual	void SetWindowHints();

	/*
	 *    Resize window.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 *    @param 		 	parameter3	The third parameter.
	 */

	virtual void ResizeWindow(GLFWwindow*, int, int);

	/*
	 *    Builds a window.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	virtual void BuildWindow(GLFWwindow*);

	/*
	 *    engages the gl context.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void PreRaster(GLFWwindow*);

	/*
	 *    disengages the gl context.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void PostRaster(GLFWwindow*);

	/*
	 *    Draws.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param [in,out]	parameter2	If non-null, the second parameter.
	 */

	virtual void Draw(GLFWwindow*, RasterJob*);

	/* Makes context current. */
	/* Makes context current. */
	void MakeContextCurrent();

	/* Information about the window. */
	/* Information about the window. */
	struct WindowInformation
	{
		/* The window */
		/* The window */
		GLFWwindow* window;
		/* Context for the gl */
		/* Context for the gl */
		std::unique_ptr<GLEWContext> glContext;
		/* The identifier */
		/* The identifier */
		uint ID;

		WindowInformation(GLFWwindow* windowIn, std::unique_ptr<GLEWContext> glContextIn) :

			/*
			 *    Constructor.
			 *
			 *    @param	parameter1	The first parameter.
			 */

			glContext(std::move(glContextIn)) {
			window = windowIn;
			ID = windowCounter++;
		}
	};


protected:

private:

	/* A gl vertex. */
	/* A gl vertex. */
	struct GLVertex
	{
		/* The position */
		/* The position */
		glm::vec4 position;
		/* The normal */
		/* The normal */
		glm::vec4 normal;
		/* The color */
		/* The color */
		glm::vec4 color;
		/* The uv */
		/* The uv */
		glm::vec2 UV;
	};

	/* The window counter */
	/* The window counter */
	static uint windowCounter;
	/* The window storage */
	/* The window storage */
	std::map<GLFWwindow*, std::unique_ptr<WindowInformation> > windowStorage;

	/*
	 *    Makes context current.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 */

	void MakeContextCurrent(GLFWwindow*);

};