#pragma once

#include <string>
#include <sstream>
#include <deque>
#include <mutex>

enum LogSeverity {
	TRACE,
	WARNING,
	ERROR,
	FATAL
};

namespace Logger {

	//User: Do Not Touch
	namespace detail {

		template <typename T>
		void LogHelp(std::ostream& o, T t)
		{
			o << t << std::endl;
		}

		template<typename T, typename... Args>
		void LogHelp(std::ostream& o, T t, Args... args)
		{
			LogHelp(o, t);
			LogHelp(o, args...);
		}

		extern std::string LogSeverityStrings[4];
		extern std::mutex logMut;


		typedef struct LogI {

			std::string msg;
			LogSeverity severity;

		}LogI;

		extern std::deque<LogI> storage;
	}

	//Logs a message of the specified type
	template<typename... Args>
	void Log(LogSeverity logType, Args... args)
	{
		std::ostringstream oss;
		detail::LogHelp(oss, args...);

		detail::logMut.lock();
		detail::storage.push_back({ oss.str() ,logType });
		detail::logMut.unlock();
	}


	//Logs a message with a TRACE type
	template<typename... Args>
	void Log(Args... args)
	{
		std::ostringstream oss;
		detail::LogHelp(oss, args...);

		detail::logMut.lock();
		detail::storage.push_back({ oss.str() ,TRACE });
		detail::logMut.unlock();
	}

	//Get the next available string to write
	std::string Get();

}