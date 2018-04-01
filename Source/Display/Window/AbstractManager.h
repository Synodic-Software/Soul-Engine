#pragma once

#include <vulkan\vulkan.hpp>
#include <GLFW\glfw3.h>

#include "Metrics.h"
#include "AbstractWindow.h"
#include "Display\Layout\Layout.h"

class AbstractManager
{
public:
	static AbstractManager& Instance() {
		static AbstractManager instance;
		return instance;
	}

	AbstractManager(AbstractManager const&) = delete;
	void operator=(AbstractManager const&) = delete;

	bool ShouldClose();
	void SignalClose();

	AbstractWindow* CreateWindow(WindowType, const std::string&, int monitor, uint x, uint y, uint width, uint height);

	void SetWindowLayout(AbstractManager*, Layout*);

	void Draw();
	void Resize(AbstractWindow*, int, int);
	void Refresh(AbstractManager*);
	void WindowPos(AbstractManager*, int, int);

	void Close(AbstractManager*);

private:

	AbstractManager();
	~AbstractManager();

	std::list<std::unique_ptr<AbstractWindow>> windows;
	AbstractWindow* masterWindow = nullptr;

	int monitorCount;
	void** monitors;

	bool runningFlag;
};