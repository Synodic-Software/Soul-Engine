#pragma once

//#include "Display/Window/Window.h"
//#include "Transput/Input/Desktop/DesktopInputManager.h"
////#include "Rasterer/RasterManager.h"
//class RasterManager;
//class EntityManage;
//
//struct GLFWmonitor;
//class DesktopWindowManager : public WindowManager
//{
//
//public:
//
//	DesktopWindowManager(EntityManager&, DesktopInputManager&, RasterManager&);
//	~DesktopWindowManager() override = default;
//	void Terminate() override;
//
//	DesktopWindowManager(const DesktopWindowManager&) = delete;
//	DesktopWindowManager(DesktopWindowManager&& o) noexcept = default;
//
//	DesktopWindowManager& operator=(const DesktopWindowManager&) = delete;
//	DesktopWindowManager& operator=(DesktopWindowManager&& other) noexcept = default;
//
//
//	// Close operations. 
//	bool ShouldClose() const override;
//	void SignalClose() override;
//
//	//Process to create a window.
//	Window& CreateWindow(WindowParameters&) override;
//
//	void Draw() override;
//
//private:
//
//	Entity masterWindow_;
//
//	EntityManager* entityManager_;
//	DesktopInputManager* inputManager_;
//	RasterManager* rasterManager_;
//
//	GLFWmonitor** monitors_;
//
//};
