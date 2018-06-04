#include "Logger.h"

namespace Logger {

	namespace detail {
		std::string LogSeverityStrings[4]{
			"TRACE",
			"WARNING",
			"ERROR",
			"FATAL"
		};

		/* The log mut */
		std::mutex logMut;
		/* The storage */
		std::deque<LogI> storage;

		/*
		 *    Writes an information.
		 *    @param [in,out]	oss 	The oss.
		 *    @param 		 	file	The file.
		 *    @param 		 	line	The line.
		 */

		void WriteInfo(std::ostream& oss, const char* file, int line) {		
			oss << "File: TODO" << " Line: " << line << " | ";
		}
	}

	/*
	 *    Gets the get.
	 *    @return	A std::string.
	 */

	std::string Get() {
		detail::logMut.lock();
		if (detail::storage.empty() > 0) {

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

	void WriteFile()
	{
		//TODO
	}
}