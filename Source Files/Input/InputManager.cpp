#include "InputManager.h"
#include "Events/EventManager.h"
#include "Multithreading/Scheduler.h"

#include <string> 

namespace InputManager {

	namespace detail {

		void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {

			//TODO cast int into properly named key -> std:string

			if (action == GLFW_PRESS) {
				keyStates[std::to_string(key)].state = PRESS;
			}
			else if (action == GLFW_REPEAT) {

			}
			else if (action == GLFW_RELEASE) {

			}
			else {
				//case GLFW_UNKNOWN
			}
		}

		void characterCallback(GLFWwindow* window, unsigned int codepoint) {

		}

		void cursorCallback(GLFWwindow* window, double xpos, double ypos) {

		}

		void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {

		}

		void buttonCallback(GLFWwindow* window, int button, int action, int mods) {

		}

		std::unordered_map<std::string, Key> keyStates;

	}

	void AttachWindow(GLFWwindow* window) {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [=]() {

			glfwSetKeyCallback(window, detail::keyCallback);
			glfwSetCharCallback(window, detail::characterCallback);
			glfwSetCursorPosCallback(window, detail::cursorCallback);
			glfwSetScrollCallback(window, detail::scrollCallback);
			glfwSetMouseButtonCallback(window, detail::buttonCallback);

		});

		Scheduler::Block();

	}

	void Poll() {

		//fire off all events
		for (auto iter : detail::keyStates) {
			EventManager::Emit("KeyInput", iter.first);
		}

		//reset any states
	}

}
