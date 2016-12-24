#pragma once

#include <string>
#include <sstream>
#include <deque>
#include <mutex>
#include <iostream>
#include <boost/filesystem.hpp>


enum LogSeverity {
	TRACE,
	WARNING,
	ERROR,
	FATAL
};

#define LOG(SEVERITY,...) Logger::detail::Log(SEVERITY,__FILE__, __LINE__, __VA_ARGS__)
#define LOG_TRACE(...) Logger::detail::Log(TRACE,__FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARNING(...) Logger::detail::Log(WARNING,__FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) Logger::detail::Log(ERROR,__FILE__, __LINE__, __VA_ARGS__)
#define LOG_FATAL(...) Logger::detail::Log(FATAL,__FILE__, __LINE__, __VA_ARGS__)

namespace Logger {

	//User: Do Not Touch
	namespace detail {

		template <typename T>
		void LogHelp(std::ostream& o, T t)
		{
			o << t;
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
			int luneNumber;
			const char* filename;

		}LogI;

		extern std::deque<LogI> storage;

		//Setters for logging messages. Interfaced by macros

		void WriteInfo(std::ostream& oss, const char* file, int line);


		//Logs a message of the specified type
		template<typename... Args>
		void Log(LogSeverity logType, const char* file, int line, Args... args)
		{
			std::ostringstream oss;
			WriteInfo(oss, file, line);
			detail::LogHelp(oss, args...);

			detail::logMut.lock();
			detail::storage.push_back({ oss.str() ,logType ,line,file });
			detail::logMut.unlock();
		}
	}

	//Get the next available string to write. Empty string if none written
	std::string Get();
}