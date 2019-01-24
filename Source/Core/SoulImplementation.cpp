#include "SoulImplementation.h"

#include "System/Platform.h"
#include "Soul.h"

#include "Transput/Input/InputManager.h"

Soul::Implementation::Implementation(Soul& soul) :
	entityManager_(),
	eventManager_(),
	inputManagerVariant_(ConstructInputManager()),
	inputManager_(ConstructInputPtr()),
	consoleManagerVariant_(ConstructConsoleManager(soul)),
	consoleManager_(ConstructConsolePtr()),
	framePipeline_(soul.schedulerModule_, {
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
	else {
		return nullptr;
	}
};
