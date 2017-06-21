#include "Logger.h"

namespace Logger {

	namespace detail {
		std::string LogSeverityStrings[4]{
			"TRACE",
			"WARNING",
			"ERROR",
			"FATAL"
		};

		std::mutex logMut;
		std::deque<LogI> storage;

		void WriteInfo(std::ostream& oss, const char* file, int line) {
			std::string baseName = boost::filesystem::path(file).filename().string();
			oss << "File: " << baseName << " Line: " << line << " | ";
		}
	}

	std::string Get() {
		detail::logMut.lock();
		if (detail::storage.size() > 0) {

			detail::LogI temp = detail::storage.front();
			detail::storage.pop_front();
			detail::logMut.unlock();

			return ("[" + detail::LogSeverityStrings[temp.severity] + "] " + temp.msg+"\n");
		}
		else {
			detail::logMut.unlock();
			return std::string();
		}

	}
}