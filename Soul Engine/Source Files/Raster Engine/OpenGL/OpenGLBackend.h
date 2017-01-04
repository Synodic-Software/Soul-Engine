#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "Display\Window\Window.h"
#include "Raster Engine\RasterBackend.h"
#include "Utility\Includes\GLMIncludes.h"

#include <vector>

typedef struct GLVertex
	{
		glm::vec4 position;
		glm::vec2 UV;
		glm::vec4 normal;
		glm::vec4 color;
	};


	typedef struct WindowInformation
	{
		Window* window;
		GLEWContext* glContext;
		unsigned int ID;
	};

class OpenGLBackend : public RasterBackend::Backend {
public:
	OpenGLBackend();
	~OpenGLBackend();

	virtual void Init();
	virtual void SCreateWindow(Window*);
	virtual void PreRaster();
	virtual void PostRaster();
	virtual void Terminate();
	virtual void Draw();

private:
	
	void MakeContextCurrent(WindowInformation*);

	uint windowCounter;
	std::vector<WindowInformation> windowStorage;
	WindowInformation* currentContext;
};