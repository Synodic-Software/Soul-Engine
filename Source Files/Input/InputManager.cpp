#include "InputManager.h"
#include "Events/EventManager.h"
#include "Multithreading/Scheduler.h"
#include "Utility/Logger.h"

#include <string> 

namespace InputManager {

	namespace detail {

		void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {

			const char* tempName = glfwGetKeyName(key, scancode);

			if (!tempName) {
				S_LOG_ERROR("glfwGetKeyName returned NULL");
			}

			std::string name = tempName;

			if (action == GLFW_PRESS) {

				//if the state was previously open, start the timer
				if (keyStates[name].state == OPEN) {
					keyStates[name].sincePress.Reset();
				}

				keyStates[name].state = PRESS;
			}
			else if (action == GLFW_REPEAT) {
				//GLFW_REPEAT is handled in GLFW, but we want custimization for input, so this is ignored

			}
			else if (action == GLFW_RELEASE) {
				keyStates[name].state = RELEASE;
			}
			else {
				//case GLFW_UNKNOWN

				S_LOG_ERROR("Reached unknown key case");

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

		//TODO break into tasks
		//fire off all events and their logic
		for (auto& iter : detail::keyStates) {

			//change state if the key timer is beyond the requested.
			if (iter.second.state == PRESS &&
				iter.second.sincePress.Elapsed() > iter.second.timeToRepeat) {

				iter.second.state == REPEAT;

			}

			EventManager::Emit("KeyInput", iter.first);

			//handle reset case after emitting a release event
			if (iter.second.state == RELEASE) {
				iter.second.state = OPEN;
			}
		}

	}

}
