#pragma once

#include "Core/Utility/CRTP.h"

template<class T>
class SoulApplication : CRTP<T, SoulApplication> {
public:


	void Initialize() {
		this->Type().Initialize();
	}




protected:

	SoulApplication()
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
bool SoulApplication<T>::registered = Register();