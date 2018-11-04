#include "SoulApplication.h"
#include "Core/Soul.h"

SoulApplication::SoulApplication(SoulParameters params) :
	hasControl(true),
	parameters(params),
	soul(parameters)
{
}

Window& SoulApplication::CreateWindow(WindowParameters& params) {

	return soul.CreateWindow(params);

}


void SoulApplication::Run() {

	CheckParameters();

	//EventManager::Listen("Input", "ESCAPE", [](keyState state) {
	//	if (state == RELEASE) {
	//		SoulSignalClose();
	//	}
	//});

	soul.Init();

}

void SoulApplication::CheckParameters() {

	//TODO
	
}
