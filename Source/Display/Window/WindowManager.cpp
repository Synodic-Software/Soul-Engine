#include "WindowManager.h"

WindowManager::WindowManager() :
	masterWindow_(nullptr),
	monitorCount_(0),
	runningFlag_(true)
{
}

void WindowManager::Draw() {
	for (auto& itr : windows_) {
		itr->Draw();
	}
}

void WindowManager::Refresh() {

}

void WindowManager::Resize(int, int) {

}

void WindowManager::WindowPos(int, int) {

}