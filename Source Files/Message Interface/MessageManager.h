//#pragma once
//
//#include "Message.h"
//#include "../Multithreading/Scheduler.h"
//
//namespace Messaging {
//	MessageManager messageHandler;
//	
//	void addMessage(MessagePointer message, FiberPolicy fiberPolicy = LAUNCH_IMMEDIATE, FiberPriority fiberPriority = FIBER_HIGH);
//	void getMessage(DestinationType dest, FiberPolicy fiberPolicy = LAUNCH_IMMEDIATE, FiberPriority fiberPriority = FIBER_HIGH);
//	void registerFunction(DestinationType dest, const std::string & message, FunctionType function, 
//		FiberPolicy fiberPolicy = LAUNCH_IMMEDIATE, FiberPriority fiberPriority = FIBER_HIGH);
//
//	class MessageManager
//	{
//	public:
//		MessageManager();
//		bool addMessage(MessagePointer message);
//		bool getMessage(DestinationType dest);
//
//		//Note: function must be implemented by engine module dest.
//		//Also, function must delete args param, or memory leaks will occur.
//		bool registerFunction(DestinationType dest, const std::string & message, FunctionType function); 
//
//	private:
//		MessageMap allMessages; //Messages to be rpocesed
//		MutexMap messageMutexes; //Mutexes to control access to messages
//		FunctionsMap allFunctions; //Functions registed by engine sub-systems
//		MutexMap functionMutexes; //Mutexes to control access to registered functions
//	};
//}