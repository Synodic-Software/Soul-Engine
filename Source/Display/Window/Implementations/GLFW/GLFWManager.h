#pragma once

#include "Display\Window\AbstractManager.h"

class GLFWManager : public AbstractManager
{
public:
	static AbstractManager& Instance() {
		static GLFWManager instance;
		return instance;
	}

	/* GLFWManager can be neither copied, nor assigned. */
	GLFWManager(GLFWManager const&) = delete;
	void operator=(GLFWManager const&) = delete;

	/* Close operations. */
	bool ShouldClose();
	void SignalClose();
	void Close(AbstractManager*);

	/* Process. to create a window. */
	AbstractWindow* CreateWindow(WindowType, const std::string&, int monitor, uint x, uint y, uint width, uint height);

	/* Set the Window's Layout. */
	void SetWindowLayout(AbstractManager*, Layout*);

	/* Modifier operations. */
	void Draw();
	void Resize(AbstractWindow*, int, int);
	void Refresh(AbstractManager*);
	void WindowPos(AbstractManager*, int, int);

private:
	/* Constructor. */
	GLFWManager();

	/* Destructor. */
	~GLFWManager();

};