#pragma once

template<class T>
class Application {
public:

	
	//virtual std::string name() = 0;
	
	static bool Initialize()
	{
		//T t;
		//AnimalManager::registerAnimal(t.name());
		return true;
	}

	static bool registered;

protected:

	Application()
	{
		registered;
	}

};

template<class T>
bool Application<T>::registered = Application<T>::Initialize();