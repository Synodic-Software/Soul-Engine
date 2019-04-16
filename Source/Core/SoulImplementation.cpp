#include "SoulImplementation.h"

#include "System/Platform.h"
#include "Soul.h"

#include "Transput/Input/InputManager.h"

Soul::Implementation::Implementation(Soul& soul) :
	entityManager_(),
	eventManager_(),
	inputManagerVariant_(ConstructInputManager()),
	inputManager_(ConstructInputPtr()),
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