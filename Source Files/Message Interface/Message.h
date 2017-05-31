//#pragma once
//
//#include "MessageTypes.h"
//
//namespace Messaging {
//
//	class Message
//	{
//	public:
//		Message(PriorityType p, DestinationType d, const std::string & c, ArgType a);
//		
//		unsigned int getId() const { return id; }
//		PriorityType getPriority() const { return priority; }
//		DestinationType getDestination() const { return destination };
//		std::string getContent() const { return content; }
//		ArgType getArgs() const { return args; }
//
//	protected:
//		PriorityType priority;
//		DestinationType destination;
//		std::string content;
//		ArgType args;
//		unsigned int id;
//
//		static std::atomic_uint32_t nextId;
//		static std::atomic_uint32_t getNextId() { return nextId++; }
//	};
//}