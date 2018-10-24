#include "SoulImplementation.h"

#include "Platform/Platform.h"
#include "Soul.h"


Soul::Implementation::Implementation(Soul& soul) :
	entityManager_(),
	scheduler_(soul.parameters.threadCount),
	eventManager_(),
	inputManagerVariant_(ConstructInputManager()),
	inputManager_(ConstructInputPtr()),
	windowManagerVariant_(ConstructWindowManager()),
	windowManager_(ConstructWindowPtr()),
	rasterManager_(scheduler_, entityManager_),
	framePipeline_(scheduler_, {
	[&soul](Frame& oldFrame, Frame& newFrame)
	{
		soul.Process(oldFrame, newFrame);
	},
	[&soul](Frame& oldFrame, Frame& newFrame)
	{
		soul.Update(oldFrame, newFrame);
	},
	[&soul](Frame& oldFrame, Frame& newFrame)
	{
		soul.Render(oldFrame, newFrame);
	} })
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
