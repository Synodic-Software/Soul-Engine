#pragma once

#include "Display\Window\AbstractManager.h"
#include "Display\Window\ManagerInterface.h"

class DesktopManager : public AbstractManager
{
public:
	/* Constructor. */
	DesktopManager();

	/* Destructor. */
	~DesktopManager() = default;
	
	/*static DesktopManager& Instance() {
		static DesktopManager instance;
		return instance;
	}*/
	
	/* GLFWManager can be neither copied, nor assigned. */
	DesktopManager(DesktopManager const&) = delete;
	void operator=(DesktopManager const&) = delete;

	/* Process to create a window. */
	AbstractWindow* CreateWindow(WindowType, const std::string&, int monitor, uint x, uint y, uint width, uint height);

	/* Set the Window's Layout. */
	void SetWindowLayout(AbstractWindow*, Layout*);

	/* Close operations. */
	bool ShouldClose();
	void SignalClose();
	void Close(void*);


	/* Modifier operations. */
	void Draw();
	void Refresh(void* handler);
	void Resize(void* handler, int width, int height);
	void WindowPos(void* handler, int x, int y);

	friend class ManagerInterface;

private:

};