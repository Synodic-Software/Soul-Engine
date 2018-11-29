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
	consoleManagerVariant_(ConstructConsoleManager(soul)),
	consoleManager_(ConstructConsolePtr()),
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

Soul::Implementation::consoleManagerVariantType Soul::Implementation::ConstructConsoleManager(Soul& soul) {

	consoleManagerVariantType tmp;

	if constexpr (Platform::WithCLI()) {
		tmp.emplace<CLIConsoleManager>(eventManager_, soul);	
	}

	return tmp;
};

ConsoleManager* Soul::Implementation::ConstructConsolePtr() {

	if constexpr (Platform::WithCLI()) {
		return &std::get<CLIConsoleManager>(consoleManagerVariant_);
	}

	return nullptr;
};
