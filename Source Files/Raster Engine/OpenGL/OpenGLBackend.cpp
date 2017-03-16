#include "OpenGLBackend.h"
#include "Utility\Logger.h"
#include "OpenGLUtility.h"
#include "Input/InputState.h"
#include "Multithreading\Scheduler.h"

uint OpenGLBackend::windowCounter = 0;
OpenGLBackend::WindowInformation* currentContext = nullptr;
OpenGLBackend::WindowInformation* defaultContext = nullptr;

//static void APIENTRY openglCallbackFunction(
//	GLenum source,
//	GLenum type,
//	GLuint id,
//	GLenum severity,
//	GLsizei length,
//	const GLchar* message,
//	const void* userParam
//) {
//	(void)source; (void)type; (void)id;
//	(void)severity; (void)length; (void)userParam;
//	fprintf(stderr, "%s\n", message);
//	if (severity == GL_DEBUG_SEVERITY_HIGH) {
//		fprintf(stderr, "Aborting...\n");
//	}
//}

//the public method that calls the private one
void  OpenGLBackend::MakeContextCurrent() {
	if (currentContext != defaultContext) {
		MakeContextCurrent(nullptr);
		MakeContextCurrent(defaultContext->window);
	}
}

//must be called on the main thread, makes the context current
void OpenGLBackend::MakeContextCurrent(GLFWwindow* window)
{
	OpenGLBackend::WindowInformation* info;
	if (window) {
		info = windowStorage.at(window).get();
		glfwMakeContextCurrent(info->window);
	}
	else {
		info = nullptr;
		glfwMakeContextCurrent(nullptr);
	}
	currentContext = info;
}

GLEWContext* glewGetContext()
{
	return currentContext->glContext.get();
}

OpenGLBackend::OpenGLBackend() {
	currentContext = nullptr;
}

OpenGLBackend::~OpenGLBackend() {

}

void OpenGLBackend::SetWindowHints(GLFWwindow*& contextIn) {
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

void OpenGLBackend::BuildWindow(GLFWwindow* window) {

	GLenum err;

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, &err, &window]() {

		windowStorage[window] = std::unique_ptr<WindowInformation>(new WindowInformation(window, std::unique_ptr<GLEWContext>(new GLEWContext())));

		if (windowStorage.size() == 1) {
			defaultContext = windowStorage[window].get();
		}

		MakeContextCurrent();

		glewExperimental = true;
		err = glewInit();

		//// Enable the debug callback
		//glEnable(GL_DEBUG_OUTPUT);
		//glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		//glDebugMessageCallback(openglCallbackFunction, nullptr);
		//glDebugMessageControl(
		//	GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, true
		//);

		if (err != 0) {
			S_LOG_FATAL(err);
		}

		while (glGetError() != GL_NO_ERROR) {}



	});

	Scheduler::Block();


	if (!windowStorage.at(window).get()->glContext.get())
	{
		S_LOG_FATAL("Could not Create GLEW OpenGL context");
	}
	if (err != GLEW_OK)
	{
		S_LOG_FATAL("GLEW Error occured, Description:", glewGetErrorString(err));
	}
}

//called by glfwPollEvents meaning no scheduling is needed for the main thread
void OpenGLBackend::ResizeWindow(GLFWwindow* window, int width, int height) {
	if (windowStorage.find(window) != windowStorage.end()) {

		MakeContextCurrent(window);
		glViewport(0, 0, width, height);
		MakeContextCurrent(nullptr);
	}
}

void OpenGLBackend::PreRaster(GLFWwindow* window) {

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

}

void OpenGLBackend::PostRaster(GLFWwindow* window) {

	glfwSwapBuffers(window);


}



void OpenGLBackend::Draw(GLFWwindow* window, RasterJob* job) {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this,&window]() {

		if (windowStorage.find(window) != windowStorage.end()) {

			MakeContextCurrent();


			PreRaster(window);



			PostRaster(window);
		}

	});

	Scheduler::Block();
}