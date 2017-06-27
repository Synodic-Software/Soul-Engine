#pragma once

#include "Utility\Includes\GLFWIncludes.h"
#include "InputSet.h"

/* . */
namespace InputManager {

	/* . */
	namespace detail {

		/*
		 *    Callback, called when the key.
		 *    @param [in,out]	window  	If non-null, the window.
		 *    @param 		 	key			The key.
		 *    @param 		 	scancode	The scancode.
		 *    @param 		 	action  	The action.
		 *    @param 		 	mods		The mods.
		 */

		void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

		/*
		 *    Callback, called when the character.
		 *    @param [in,out]	window   	If non-null, the window.
		 *    @param 		 	codepoint	The codepoint.
		 */

		void characterCallback(GLFWwindow* window, unsigned int codepoint);

		/*
		 *    Callback, called when the cursor.
		 *    @param [in,out]	window	If non-null, the window.
		 *    @param 		 	xpos  	The xpos.
		 *    @param 		 	ypos  	The ypos.
		 */

		void cursorCallback(GLFWwindow* window, double xpos, double ypos);

		/*
		 *    Callback, called when the scroll.
		 *    @param [in,out]	window 	If non-null, the window.
		 *    @param 		 	xoffset	The xoffset.
		 *    @param 		 	yoffset	The yoffset.
		 */

		void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

		/*
		 *    Callback, called when the button.
		 *    @param [in,out]	window	If non-null, the window.
		 *    @param 		 	button	The button.
		 *    @param 		 	action	The action.
		 *    @param 		 	mods  	The mods.
		 */

		void buttonCallback(GLFWwindow* window, int button, int action, int mods);
	}

	/*
	 *    Attach window.
	 *    @param [in,out]	window	If non-null, the window.
	 */

	void AttachWindow(GLFWwindow* window);
};
