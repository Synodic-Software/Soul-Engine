#include "InputManager.h"
#include "Events/EventManager.h"
#include "Multithreading/Scheduler.h"

#include <string> 

namespace InputManager {

	namespace detail {

		void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
			EventManager::Emit("KeyInput", std::to_string(key));
		}

		void characterCallback(GLFWwindow* window, unsigned int codepoint) {
			
		}

		void cursorCallback(GLFWwindow* window, double xpos, double ypos) {
			
		}

		void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
			
		}

		void buttonCallback(GLFWwindow* window, int button, int action, int mods) {
			
		}

		std::pair<keyState, keyInfo> keyStates[348];

	}

	void AttachWindow(GLFWwindow* window) {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [=]() {

			glfwSetKeyCallback(window,detail::keyCallback);
			glfwSetCharCallback(window, detail::characterCallback);
			glfwSetCursorPosCallback(window, detail::cursorCallback);
			glfwSetScrollCallback(window, detail::scrollCallback);
			glfwSetMouseButtonCallback(window, detail::buttonCallback);

		});

		Scheduler::Block();

	}

	void Poll() {
		
	}

}
