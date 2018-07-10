#pragma once

#include "Display/Window/Window.h"
#include "Transput/Input/Desktop/DesktopInputManager.h"

class DesktopWindow : public Window 
{
public:

	DesktopWindow(WindowParameters&, GLFWmonitor*, DesktopInputManager&, EntityManager&);
	~DesktopWindow() override;

	DesktopWindow(const DesktopWindow &) = delete;
	DesktopWindow(DesktopWindow &&) noexcept = default;

	DesktopWindow& operator=(const DesktopWindow &) = delete;
	DesktopWindow& operator=(DesktopWindow &&) noexcept = default;

	void Draw() override;

	void Refresh() override;
	void Close() override;
	void Resize(int, int) override;
	void PositionUpdate(int,int) override;
	void SetLayout(Layout*) override;

	DesktopInputManager& GetInputSet() const;

private:

	//the stored input manager will always be desktop, as this class is a the desktop version
	DesktopInputManager* inputManager_;

};