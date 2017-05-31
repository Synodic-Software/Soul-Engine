//#include "MessageManager.h"
//
//namespace Messaging {
//	MessageManager::MessageManager() {
//		for (int i = 0; i < detail::destinations.size(); ++i) {
//			DestinationType dest = detail::destinations[i];
//			allMessages[dest]; // = MessageQueue(); // create a message queue for each destination
//			messageMutexes[dest]; // Default construct a mutex
//			allFunctions[dest];
//			functionMutexes[dest];
//		}
//	}
//
//	bool MessageManager::addMessage(MessagePointer message) {
//		if (!message) {
//			delete message;
//			return false;
//		}
//		
//		DestinationType dest = message->getDestination();
//		PriorityType p = message->getPriority();
//		messageMutexes[dest].lock();
//		MessageQueue & queue = allMessages[dest];
//		
//		queue.push(message);
//		messageMutexes[dest].unlock();
//
//		if (p == IMMEDIATE) {
//			//Asynchronous
//			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, false, [dest,this]() {
//				this->getMessage(dest);
//			});
//
//			//Synchronous
//			//this->getMessage(dest);
//		}
//
//		return true;
//	}
//
//	bool MessageManager::getMessage(DestinationType dest) {
//		messageMutexes[dest].lock();
//		MessageQueue & queue = allMessages[dest];
//		if (queue.empty()) {
//			messageMutexes[dest].unlock();
//			return false;
//		}
//
//		MessagePointer message = queue.top();
//		queue.pop();
//		messageMutexes[dest].unlock();
//
//		//unpack message
//		std::string content = message->getContent();
//		ArgType args = message->getArgs();
//		PriorityType priority = message->getPriority();
//
//		delete message;
//
//		FiberPolicy policy = (priority == IMMEDIATE) ? LAUNCH_IMMEDIATE : LAUNCH_CONTINUE;
//		FiberPriority fiberPriority = (priority == LOW) ? FIBER_LOW : FIBER_HIGH;
//
//		functionMutexes[dest].lock();
//		if (allFunctions[dest].find(content) == allFunctions[dest].end()) return false; //No Function registered for this content -- Could log that content not registered here
//		std::function<void(ArgType)> function = allFunctions[dest][content];
//		functionMutexes[dest].unlock();
//
//		Scheduler::AddTask(policy, fiberPriority, false, [function, args]() {
//			function(args);
//		});
//
//		// PROCESS MESSAGE HERE
//		/*
//			Lock Functions map mutex
//			Get corresponding function
//			Unlock Functions map
//			Cast ArgType to correct pointer -- do this in destination function??
//			Send function to scheduler with givern args as lambda function with
//				args being captured
//			
//			NOTE: this requires each engine module to implement their own functions to pass
//		*/
//
//		return true;
//	}
//
//	bool MessageManager::registerFunction(DestinationType dest, const std::string & message, FunctionType function) {
//		bool overwritten = false;
//		functionMutexes[dest].lock();
//
//		FunctionMap & thisMap = allFunctions[dest];
//		if (thisMap.find(message) != thisMap.end()) overwritten = true; //Overwriting existing registration
//		thisMap[message] = function;
//		
//		functionMutexes[dest].unlock();
//		return overwritten;
//	}
//
//	void addMessage(MessagePointer message, FiberPolicy fiberPolicy, FiberPriority fiberPriority) {
//		Scheduler::AddTask(fiberPolicy, fiberPriority, false, [message](){
//			messageHandler.addMessage(message);
//		});
//	}
//
//	void getMessage(DestinationType dest, FiberPolicy fiberPolicy, FiberPriority fiberPriority) {
//		Scheduler::AddTask(fiberPolicy, fiberPriority, false, [dest](){
//			messageHandler.getMessage(dest);
//		});
//	}
//
//	void registerFunction(DestinationType dest, const std::string & message, FunctionType function,
//		FiberPolicy fiberPolicy, FiberPriority fiberPriority) {
//
//		Scheduler::AddTask(fiberPolicy, fiberPriority, false, [dest,message,function](){
//			messageHandler.registerFunction(dest, message, function);
//		});
//	}
//}