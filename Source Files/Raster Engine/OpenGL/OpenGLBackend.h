#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBase.h"
#include "Utility\Includes\GLMIncludes.h"

#include <map>
#include <memory>

GLEWContext* glewGetContext();

class OpenGLBackend : public RasterBase {
public:
	OpenGLBackend();
	~OpenGLBackend();

	virtual GLFWwindow* GetResourceContext();
	virtual	void SetWindowHints();
	virtual void ResizeWindow(GLFWwindow*, int, int);
	virtual void BuildWindow(GLFWwindow*);

	//engages the gl context
	void PreRaster(GLFWwindow*);

	//disengages the gl context
	void PostRaster(GLFWwindow*);
	virtual void Draw(GLFWwindow*, RasterJob*);

	void MakeContextCurrent();

	struct WindowInformation
	{
		GLFWwindow* window;
		std::unique_ptr<GLEWContext> glContext;
		uint ID;

		WindowInformation(GLFWwindow* windowIn, std::unique_ptr<GLEWContext> glContextIn) :
			glContext(std::move(glContextIn)) {
			window = windowIn;
			ID = windowCounter++;
		}
	};


protected:

private:

	struct GLVertex
	{
		glm::vec4 position;
		glm::vec4 normal;
		glm::vec4 color;
		glm::vec2 UV;
	};

	static uint windowCounter;
	std::map<GLFWwindow*, std::unique_ptr<WindowInformation> > windowStorage;

	void MakeContextCurrent(GLFWwindow*);

};