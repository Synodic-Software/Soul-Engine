#include "OpenGLBackend.h"
#include "Utility\Logger.h"
#include "OpenGLUtility.h"
#include "Utility\Includes\GLMIncludes.h"
#include "Input/InputState.h"


typedef struct GLVertex
{
	glm::vec4 position;
	glm::vec2 UV;
	glm::vec4 normal;
	glm::vec4 color;
};


typedef struct WindowInformation
{
	Window*  window;
	GLEWContext* glContext;
	glm::mat4    projection;
	glm::mat4    view;
	unsigned int ID;
};

unsigned int windowCounter = 0;
std::list<WindowInformation*> windows;
WindowInformation* currentContext = nullptr;


void MakeContextCurrent(WindowInformation* inWindow)
{
	if (inWindow != nullptr)
	{
		glfwMakeContextCurrent(inWindow->window->windowHandle);
		currentContext = inWindow;
	}
}

void GLFWWindowSizeCallback(GLFWwindow* inWindow, int inWidth, int inHeight)
{
	// find the WindowInformation data corrosponding to inWindow;
	WindowInformation* info = nullptr;

	for (auto& itr : windows)
	{
		if (itr->window->windowHandle == inWindow)
		{
			info = itr;
			info->projection = glm::perspective(45.0f, float(inWidth) / float(inHeight), 0.1f, 1000.0f);
		}
	}

	MakeContextCurrent(currentContext);
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

}

OpenGLBackend::~OpenGLBackend() {

}

void OpenGLBackend::Init() {

}

void OpenGLBackend::CreateWindow(Window* window, GLFWmonitor* monitor, GLFWwindow* sharedContextin) {

	WindowInformation* hPreviousContext = currentContext;

	GLFWwindow* sharedContext;
	if (sharedContextin != nullptr) 
	{
		sharedContext = sharedContextin;
	}
	else
	{
		sharedContext = nullptr;
	}

	WindowInformation* newWindow = new WindowInformation;

	newWindow->glContext = nullptr;
	newWindow->window = window;
	newWindow->ID = windowCounter++; 


	WindowType  win = BORDERLESS;

	glfwWindowHint(GLFW_SAMPLES, 0);
	glfwWindowHint(GLFW_VISIBLE, true);

	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	if (win == FULLSCREEN) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

		glfwWindowHint(GLFW_RED_BITS, mode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		newWindow->window->windowHandle = glfwCreateWindow(window->width, window->height, window->title.c_str(), monitor, sharedContext);

	}
	else if (win == WINDOWED) {

		glfwWindowHint(GLFW_RESIZABLE, true);
		newWindow->window->windowHandle = glfwCreateWindow(window->width, window->height, window->title.c_str(), nullptr, sharedContext);

	}

	else if (win == BORDERLESS) {

		glfwWindowHint(GLFW_RESIZABLE, false);
		glfwWindowHint(GLFW_DECORATED, false);

		glfwWindowHint(GLFW_RED_BITS, mode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		newWindow->window->windowHandle = glfwCreateWindow(window->width, window->height, window->title.c_str(), nullptr, sharedContext);

	}
	else {
		LOG(ERROR, "No Window setting found");
	}



	if (newWindow->window->windowHandle == nullptr)
	{
		LOG(ERROR, "Could not Create GLFW Window");
		delete newWindow;
		return;
	}

	glfwSetWindowPos(newWindow->window->windowHandle, window->xPos, window->yPos);

	newWindow->glContext = new GLEWContext();
	if (newWindow->glContext == nullptr)
	{
		LOG(ERROR, "Could not Create GLEW OpenGL context");
		delete newWindow;
		return;
	}

	MakeContextCurrent(newWindow);

	GLenum err = glewInit();

	if (err != GLEW_OK)
	{
		LOG(ERROR, "GLEW Error occured, Description:", glewGetErrorString(err));
		glfwDestroyWindow(newWindow->window->windowHandle);
		delete newWindow;
		return;
	}

	glfwSetWindowSizeCallback(newWindow->window->windowHandle, GLFWWindowSizeCallback);

	windows.push_back(newWindow);

	//glfwSetWindowUserPointer(windowHandle, this);

	glfwSetKeyCallback(newWindow->window->windowHandle, Input::KeyCallback);
	glfwSetScrollCallback(newWindow->window->windowHandle, Input::ScrollCallback);
	glfwSetCursorPosCallback(newWindow->window->windowHandle, Input::MouseCallback);

	// now restore previous context:
	//MakeContextCurrent(hPreviousContext);

}

void OpenGLBackend::PreRaster() {

}

void OpenGLBackend::PostRaster() {

}

void OpenGLBackend::Terminate() {

}