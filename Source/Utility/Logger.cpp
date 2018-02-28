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
			std::string baseName = boost::filesystem::path(file).filename().string();
			oss << "File: " << baseName << " Line: " << line << " | ";
		}
	}

	/*
	 *    Gets the get.
	 *    @return	A std::string.
	 */

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

	//limited to a set "test default" output, overwrites existing log file
	void WriteFile()
	{
		namespace pt = boost::posix_time;
		std::ostringstream msg;
		std::string time;
		std::ofstream log("Engine.log");
		//should be changed to pull from a storage system
		char* file = "Engine.log";
		//should be changed like above
		int line;
		line = 7;
		char* str;
		time = pt::to_iso_string(pt::second_clock::universal_time());
		log << time << std::endl;
		detail::WriteInfo(log, file, line);
		log.close();
	}
}