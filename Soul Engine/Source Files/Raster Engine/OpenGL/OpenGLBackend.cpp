#include "OpenGLBackend.h"
#include "Utility\Logger.h"
#include "OpenGLUtility.h"
#include "Input/InputState.h"
#include "Multithreading\Scheduler.h"
#include "Utility\Includes\GLMIncludes.h"

struct GLVertex
{
	glm::vec4 position;
	glm::vec2 UV;
	glm::vec4 normal;
	glm::vec4 color;
};


struct WindowInformation
{
	GLFWwindow* window;
	GLEWContext* glContext;
	unsigned int ID;
};

static uint windowCounter;
static std::vector<WindowInformation> windowStorage;
static WindowInformation* currentContext;

//must be called on the main thread
void MakeContextCurrent(WindowInformation* inWindow)
{
	if (inWindow) {
		glfwMakeContextCurrent(inWindow->window);
	}
	else {
		glfwMakeContextCurrent(nullptr);
	}
	currentContext = inWindow;
}

GLEWContext* glewGetContext()
{
	return currentContext->glContext;
}

void CheckForGLErrors(std::string mssg)
{
	GLenum error = glGetError();
	while (error != GL_NO_ERROR)  // make sure we check all Error flags!
	{
		printf("Error: %s, ErrorID: %i: %s\n", mssg.c_str(), error, gluErrorString(error));
		error = glGetError(); // get next error if any.
	}
}

OpenGLBackend::OpenGLBackend() {
	currentContext = nullptr;
	windowCounter = 0;
}

OpenGLBackend::~OpenGLBackend() {

}

void OpenGLBackend::Init() {

}

void OpenGLBackend::BuildWindow(GLFWwindow* window) {



	WindowInformation newWindow{};

	newWindow.window = window;
	newWindow.ID = windowCounter++;

	GLenum err;

	//Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, window, &newWindow, &err]() {

		newWindow.glContext = new GLEWContext();
	
		MakeContextCurrent(&newWindow);
		glewExperimental = true;
		err = glewInit();

		MakeContextCurrent(nullptr);
//	});

	//Scheduler::Block();

	//add the new reference with encapsulating data
	windowStorage.push_back(newWindow);

	if (!newWindow.glContext)
	{
		S_LOG_FATAL("Could not Create GLEW OpenGL context");
	}
	if (err != GLEW_OK)
	{
		S_LOG_FATAL("GLEW Error occured, Description:", glewGetErrorString(err));
	}
}

void OpenGLBackend::PreRaster() {

	for (auto& winInfo : windowStorage) {
		GLFWwindow* handler = winInfo.window;
		//Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&]() {

			MakeContextCurrent(&winInfo);
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			MakeContextCurrent(nullptr);

		//});
	}
	//Scheduler::Block();

}

void OpenGLBackend::PostRaster() {

}

void OpenGLBackend::Terminate() {

	for (auto& winInfo : windowStorage) {
		delete winInfo.glContext;
	}
}

void OpenGLBackend::Draw() {

	for (auto& winInfo : windowStorage) {
		GLFWwindow* handler = winInfo.window;
		//Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&]() {

			MakeContextCurrent(&winInfo);
			glfwSwapBuffers(handler);
			MakeContextCurrent(nullptr);

	//	});
	}

	//Scheduler::Block();
}