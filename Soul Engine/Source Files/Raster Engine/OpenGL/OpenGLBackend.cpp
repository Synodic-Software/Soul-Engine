#include "OpenGLBackend.h"
#include "Utility\Logger.h"
#include "OpenGLUtility.h"
#include "Input/InputState.h"
#include "Multithreading\Scheduler.h"
#include "Utility\Includes\GLMIncludes.h"

#include <map>

struct GLVertex
{
	glm::vec4 position;
	glm::vec2 UV;
	glm::vec4 normal;
	glm::vec4 color;
};

static uint windowCounter;
static std::map<GLFWwindow*, OpenGLBackend::WindowInformation> windowStorage;
static OpenGLBackend::WindowInformation* currentContext;

//must be called on the main thread
void OpenGLBackend::MakeContextCurrent(WindowInformation* window)
{
	if (window) {
		glfwMakeContextCurrent(window->window);
	}
	else {
		glfwMakeContextCurrent(nullptr);
	}
	currentContext = window;
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

void OpenGLBackend::SetWindowHints() {
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

void OpenGLBackend::BuildWindow(GLFWwindow* window) {



	WindowInformation newWindow{};

	newWindow.window = window;
	newWindow.ID = windowCounter++;

	GLenum err;

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&]() {

		newWindow.glContext = new GLEWContext();

		MakeContextCurrent(&newWindow);
		glewExperimental = true;
		err = glewInit();

		MakeContextCurrent(nullptr);
	});

	Scheduler::Block();

	//add the new reference with encapsulating data
	windowStorage.insert(std::make_pair(window, newWindow));

	if (!newWindow.glContext)
	{
		S_LOG_FATAL("Could not Create GLEW OpenGL context");
	}
	if (err != GLEW_OK)
	{
		S_LOG_FATAL("GLEW Error occured, Description:", glewGetErrorString(err));
	}
}

//called by glfwPollEvents meaning no scheduling is needed for the main thread
void OpenGLBackend::ResizeWindow(GLFWwindow* window, int inWidth, int inHeight) {
	MakeContextCurrent(&windowStorage.at(window));
	glViewport(0, 0, inWidth, inHeight);
	MakeContextCurrent(nullptr);
}

void OpenGLBackend::PreRaster(GLFWwindow* window) {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, window]() {
			MakeContextCurrent(&windowStorage.at(window));
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			MakeContextCurrent(nullptr);
		});

	Scheduler::Block();

}

void OpenGLBackend::PostRaster(GLFWwindow*) {

}

void OpenGLBackend::Terminate() {

	for (auto& winInfo : windowStorage) {
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, [this, winInfo]() {
			delete winInfo.second.glContext;
		});
	}
	Scheduler::Block();

	windowStorage.clear();
}

void OpenGLBackend::Draw(GLFWwindow* inWindow) {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, inWindow]() {
		MakeContextCurrent(&windowStorage.at(inWindow));
		glfwSwapBuffers(inWindow);
		MakeContextCurrent(nullptr);
	});

	Scheduler::Block();
}