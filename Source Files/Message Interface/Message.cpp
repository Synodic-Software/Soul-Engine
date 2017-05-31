//#include "Message.h"
//
//namespace Messaging {
//	std::atomic_uint32_t Message::nextId = 0;
//
//	Message::Message(PriorityType p, Destination d, const std::string& c, ArgType a)
//		: priority(p), destination(d), args(a) {
//		
//		id = static_cast<unsigned int>(getNextId());
//		content = std::string(c);
//	}
//}