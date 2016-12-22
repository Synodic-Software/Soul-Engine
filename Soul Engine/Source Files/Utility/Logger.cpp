#include "Logger.h"

namespace Logger {

	namespace detail{
		std::string LogSeverityStrings[4]{
			"TRACE",
			"WARNING",
			"ERROR",
			"FATAL"
		};

		std::mutex logMut;
		std::deque<LogI> storage;
	}

	std::string Get() {
		detail::logMut.lock();
		detail::LogI temp = detail::storage.front();
		detail::storage.pop_front();
		detail::logMut.unlock();

		return detail::LogSeverityStrings[temp.severity] +": "+temp.msg;
		
	}
}