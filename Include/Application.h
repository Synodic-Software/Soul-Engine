#pragma once

#include "Core/Utility/CRTP.h"

template<class T>
class Application : CRTP<T, Application> {
public:


	void Initialize() {
		this->Type().Initialize();
	}




protected:

	Application()
	{
		registered = false;	//forces static specialization
	}

	bool setupWindow;

private:

	bool Register()
	{
		//AnimalManager::registerAnimal(t.name());
		return true;
	}

	static bool registered;

};

template<class T>
bool Application<T>::registered = Register();