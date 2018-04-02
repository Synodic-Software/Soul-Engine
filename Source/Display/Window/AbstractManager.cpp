#pragma once
#include "AbstractWindow.h"
#include "AbstractManager.h"

#include <vulkan\vulkan.hpp>
#include <GLFW\glfw3.h>

#include "Metrics.h"
#include "Display\Layout\Layout.h"

AbstractManager::AbstractManager()
{
	monitorCount = 0;
	monitors = nullptr;
	runningFlag = true;

}

AbstractManager::~AbstractManager() {
	masterWindow = nullptr;
}

void AbstractManager::Draw() {
	for (auto& itr : windows) {
		itr->Draw();
	}
}