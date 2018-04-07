#pragma once
#include "AbstractManager.h"

class CentralManager
{
public:
	static CentralManager& Instance() {
		static CentralManager instance;
		return instance;
	}

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
	void Resize(void*, int, int);
	void Refresh(void*);
	void WindowPos(void*, int, int);


private:
	/* Constructor. */
	CentralManager();

	/* Destructor. */
	~CentralManager();

	/* Pointer to the manager. */
	AbstractManager* manager;
};