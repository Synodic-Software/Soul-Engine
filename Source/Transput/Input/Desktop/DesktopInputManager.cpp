#include "DesktopInputManager.h"
#include "Composition/Event/EventManager.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Core/Utility/Log/Logger.h"
#include "Platform/Platform.h"

DesktopInputManager::DesktopInputManager(EventManager& eventManager) :
	InputManager(eventManager),
	mouseXOffset_(0),
	mouseYOffset_(0),
	firstMouse_(true)
{
}

void DesktopInputManager::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	if (action == GLFW_PRESS) {

		//if the state was previously open, start the timer
		if (keyStates_[key].state == KeyState::OPEN) {
			keyStates_[key].sincePress.Reset();
		}

		keyStates_[key].state = KeyState::PRESS;
	}
	else if (action == GLFW_REPEAT) {
		//GLFW_REPEAT is handled in GLFW, but we want custimization for input, so this is ignored
	}
	else if (action == GLFW_RELEASE) {
		keyStates_[key].state = KeyState::RELEASE;
	}
	else {
		//case GLFW_UNKNOWN
		// TODO: proper error handling
		S_LOG_ERROR("Reached unknown key case");
	}

}

void DesktopInputManager::CharacterCallback(GLFWwindow* window, uint codepoint) {

}

void DesktopInputManager::ModdedCharacterCallback(GLFWwindow* window, uint, int) {

}

void DesktopInputManager::ButtonCallback(GLFWwindow* window, int button, int action, int mods) {

}

void DesktopInputManager::CursorCallback(GLFWwindow* window, double xPos, double yPos) {

	if (firstMouse_) {
		mouseXPos_ = xPos;
		mouseYPos_ = yPos;
		firstMouse_ = false;
	}

	mouseXOffset_ = xPos - mouseXPos_;
	mouseYOffset_ = yPos - mouseYPos_;
	mouseXPos_ = xPos;
	mouseYPos_ = yPos;

}

void DesktopInputManager::CursorEnterCallback(GLFWwindow* window, int) {

}
void DesktopInputManager::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {

}


//void DesktopInputManager::AttachWindow(DesktopWindow* window) {
//
//	//If the desktop is used, glfw is present in both cases
//	const auto context_ = std::any_cast<GLFWwindow*>(window->context_);
//
//	//TODO construct templated function for all callbacks
//	//register the input with the Window
//	
//	glfwSetKeyCallback(context_, [](GLFWwindow* window, int key, int scancode, int action, int mods)
//	{
//		auto thisManager = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(window))->GetInputSet();
//		thisManager.KeyCallback(window, key, scancode, action, mods);
//	});
//
//	glfwSetCharCallback(context_, [](GLFWwindow* window, uint codepoint)
//	{
//		auto thisManager = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(window))->GetInputSet();
//		thisManager.CharacterCallback(window, codepoint);
//	});
//
//	glfwSetCharModsCallback(context_, [](GLFWwindow* window, uint a, int b)
//	{
//		auto thisManager = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(window))->GetInputSet();
//		thisManager.ModdedCharacterCallback(window, a, b);
//	});
//
//	glfwSetMouseButtonCallback(context_, [](GLFWwindow* window, int button, int action, int mods)
//	{
//		auto thisManager = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(window))->GetInputSet();
//		thisManager.ButtonCallback(window, button, action, mods);
//	});
//
//	glfwSetCursorPosCallback(context_, [](GLFWwindow* window, double xPos, double yPos)
//	{
//		auto thisManager = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(window))->GetInputSet();
//		thisManager.CursorCallback(window, xPos, yPos);
//	});
//
//	glfwSetCursorEnterCallback(context_, [](GLFWwindow* window, int temp)
//	{
//		auto thisManager = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(window))->GetInputSet();
//		thisManager.CursorEnterCallback(window, temp);
//	});
//
//	glfwSetScrollCallback(context_, [](GLFWwindow* window, double xoffset, double yoffset)
//	{
//		auto thisManager = static_cast<DesktopWindow*>(glfwGetWindowUserPointer(window))->GetInputSet();
//		thisManager.ScrollCallback(window, xoffset, yoffset);
//	});
//
//}

bool DesktopInputManager::Poll() {

	//process all events (includes Window events)
	glfwPollEvents();

	//TODO break into tasks 
	//fire off all key events and their logic
	for (auto&[keyID, key] : keyStates_) {

		if (key.state != KeyState::OPEN) {

			//change state if the key timer is beyond the requested.
			if (key.state == KeyState::PRESS &&
				key.sincePress.Elapsed() > key.timeToRepeat) {

				key.state = KeyState::REPEAT;

			}

			//TODO implement inputsets
			const auto inputSetOffset = 0 * 350; //350+ are not defined in GLFW keystates
			eventManager_->Emit("Input"_hashed, keyID + inputSetOffset, key.state);

			//handle reset case after emitting a release event
			if (key.state == KeyState::RELEASE) {
				key.state = KeyState::OPEN;
			}

		}
	}

	//emit mouse tasks
	eventManager_->Emit("Input"_hashed, "Mouse Position"_hashed, mouseXOffset_, mouseYOffset_);

	//zero offsets
	mouseXOffset_ = 0.0;
	mouseYOffset_ = 0.0;

	return true;
}
