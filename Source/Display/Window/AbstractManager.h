#pragma once

#include <vulkan\vulkan.hpp>
#include <GLFW\glfw3.h>

#include "Metrics.h"
#include "AbstractWindow.h"
#include "Display\Layout\Layout.h"

class AbstractManager
{
public:

	/* 
		Creates a static AbstractManager instance. 
		Only one instance of the AbstractManager should exist at a time. 
	*/
	static AbstractManager& Instance() {
		static AbstractManager instance;
		return instance;
	}

	/* AbstractManager can be neither copied, nor assigned. */
	AbstractManager(AbstractManager const&) = delete;
	void operator=(AbstractManager const&) = delete;


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
	AbstractManager();
	/* Destructor. */
	~AbstractManager();

	/* List of all windows handled by the AbstractManager. */
	std::list<std::unique_ptr<AbstractWindow>> windows;
	/* Pointer to the master--or 'main'--window. */
	AbstractWindow* masterWindow = nullptr;

	/* 
		Monitor Count Variables. 
		monitors is a void* for abstraction purposes.
	*/
	int monitorCount;
	void** monitors;

	/* Flag that keeps track of if the AbstractManager is running. */
	bool runningFlag;
};