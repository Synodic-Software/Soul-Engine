#pragma once
#include "Soul.h"
#include "Platform/Platform.h"

#include "Display/Window/WindowManager.h"
#include "Display/Window/Desktop/DesktopWindowManager.h"

#include "Transput/Input/Desktop/DesktopInputManager.h"

#include "Composition/Event/EventManager.h"
#include "Transput/Input/InputManager.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Core/Utility/HashString/HashString.h"
#include "Composition/Entity/EntityManager.h"
#include "Rasterer/RasterManager.h"

#include <variant>

class Soul::Implementation
{

public:

	//monostate allows for empty construction
	using inputManagerVariantType = std::variant<std::monostate, DesktopInputManager>;
	using windowManagerVariantType = std::variant<std::monostate, DesktopWindowManager>;

	Implementation(const Soul&);
	~Implementation();

	//services and modules	
	EntityManager entityManager_;
	Scheduler scheduler_;
	EventManager eventManager_;
	inputManagerVariantType inputManagerVariant_;
	InputManager* inputManager_;
	windowManagerVariantType windowManagerVariant_;
	WindowManager* windowManager_;
	RasterManager rasterManager_;

	FrameManager frameManager;

private:

	inputManagerVariantType ConstructInputManager();
	InputManager* ConstructInputPtr();

	windowManagerVariantType ConstructWindowManager();
	WindowManager* ConstructWindowPtr();
};

Soul::Implementation::Implementation(const Soul& soul) :
	entityManager_(),
	scheduler_(soul.parameters.threadCount),
	eventManager_(),
	inputManagerVariant_(ConstructInputManager()),
	inputManager_(ConstructInputPtr()),
	windowManagerVariant_(ConstructWindowManager()),
	windowManager_(ConstructWindowPtr()),
	rasterManager_(scheduler_, entityManager_),
	frameManager()
{
}

Soul::Implementation::~Implementation() {
	windowManager_->Terminate();
}

Soul::Implementation::inputManagerVariantType Soul::Implementation::ConstructInputManager() {

	inputManagerVariantType tmp;

	if constexpr (Platform::IsDesktop()) {
		tmp.emplace<DesktopInputManager>(eventManager_);
		return tmp;
	}

}

InputManager* Soul::Implementation::ConstructInputPtr() {

	if constexpr (Platform::IsDesktop()) {
		return &std::get<DesktopInputManager>(inputManagerVariant_);
	}

}

Soul::Implementation::windowManagerVariantType Soul::Implementation::ConstructWindowManager() {

	windowManagerVariantType tmp;

	if constexpr (Platform::IsDesktop()) {
		tmp.emplace<DesktopWindowManager>(entityManager_, std::get<DesktopInputManager>(inputManagerVariant_), rasterManager_);
		return tmp;
	}

}

WindowManager* Soul::Implementation::ConstructWindowPtr() {

	if constexpr (Platform::IsDesktop()) {
		return &std::get<DesktopWindowManager>(windowManagerVariant_);
	}
}
