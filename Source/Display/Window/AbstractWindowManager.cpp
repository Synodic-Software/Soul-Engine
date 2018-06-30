#include "AbstractWindowManager.h"
#include "System/System.h"
#include "Desktop/DesktopWindowManager.h"

AbstractWindowManager::AbstractWindowManager() :
	masterWindow(nullptr),
	monitorCount(0),
	runningFlag(true)
{
}

AbstractWindowManager::~AbstractWindowManager() {
	masterWindow = nullptr;
}

void AbstractWindowManager::Draw() {
	for (auto& itr : windows) {
		itr->Draw();
	}
}

void AbstractWindowManager::Refresh() {

}

void AbstractWindowManager::Resize(int, int) {

}

void AbstractWindowManager::WindowPos(int, int) {

}

//Factory Definitions
std::unique_ptr<AbstractWindowManager> WindowManagerFactory::CreateWindowManager() const{

	if constexpr (System::IsDesktop()) {
			return std::make_unique<DesktopWindowManager>();
	}

	return nullptr;

}