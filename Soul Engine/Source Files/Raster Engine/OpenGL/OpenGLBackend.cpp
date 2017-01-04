#include "OpenGLBackend.h"
#include "Utility\Logger.h"
#include "OpenGLUtility.h"
#include "Input/InputState.h"
#include "Multithreading\Scheduler.h"

void OpenGLBackend::MakeContextCurrent(WindowInformation* inWindow)
{
	if (inWindow != nullptr) {
		glfwMakeContextCurrent(inWindow->window->windowHandle);
	}
	currentContext = inWindow;
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

void OpenGLBackend::SCreateWindow(Window* window) {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, window]() {

		WindowInformation* previousContext = currentContext;

		WindowInformation newWindow;

		newWindow.glContext = nullptr;
		newWindow.window = window;
		newWindow.ID = windowCounter++;
		newWindow.glContext = new GLEWContext();
		MakeContextCurrent(&newWindow);
		GLenum err = glewContextInit(newWindow.glContext);

		if (newWindow.glContext == nullptr)
		{
			S_LOG_FATAL("Could not Create GLEW OpenGL context");
		}
		if (err != GLEW_OK)
		{
			S_LOG_FATAL("GLEW Error occured, Description:", glewGetErrorString(err));
		}

		//add the new reference with encapsulating data
		windowStorage.push_back(newWindow);

		MakeContextCurrent(previousContext);
	});

	Scheduler::Block();
}

void OpenGLBackend::PreRaster() {

	for (auto& winInfo : windowStorage) {
		GLFWwindow* handler = winInfo.window->windowHandle;
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&]() {

			WindowInformation* previousContext = currentContext;
			MakeContextCurrent(&winInfo);
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			MakeContextCurrent(previousContext);

		});
	}
	Scheduler::Block();

}

void OpenGLBackend::PostRaster() {

}

void OpenGLBackend::Terminate() {

}

void OpenGLBackend::Draw() {

	for (auto& winInfo : windowStorage) {
		GLFWwindow* handler = winInfo.window->windowHandle;
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&]() {

			WindowInformation* previousContext = currentContext;
			MakeContextCurrent(&winInfo);
			glfwSwapBuffers(handler);
			MakeContextCurrent(previousContext);

		});
	}

	Scheduler::Block();
}