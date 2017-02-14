#pragma once

#include <string>
#include <sstream>
#include <deque>
#include <mutex>
#include <iostream>
#include <boost/filesystem.hpp>


enum SLogSeverity {
	S_TRACE,
	S_WARNING,
	S_ERROR,
	S_FATAL
};

#define S_LOG(SEVERITY,...) Logger::detail::Log(SEVERITY,__FILE__, __LINE__, __VA_ARGS__)
#define S_LOG_TRACE(...) Logger::detail::Log(S_TRACE,__FILE__, __LINE__, __VA_ARGS__)
#define S_LOG_WARNING(...) Logger::detail::Log(S_WARNING,__FILE__, __LINE__, __VA_ARGS__)
#define S_LOG_ERROR(...) Logger::detail::Log(S_ERROR,__FILE__, __LINE__, __VA_ARGS__)
#define S_LOG_FATAL(...) Logger::detail::Log(S_FATAL,__FILE__, __LINE__, __VA_ARGS__)

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
			SLogSeverity severity;
			int lineNumber;
			const char* filename;

		}LogI;

		extern std::deque<LogI> storage;

		//Setters for logging messages. Interfaced by macros

		void WriteInfo(std::ostream& oss, const char* file, int line);


		//Logs a message of the specified type
		template<typename... Args>
		void Log(SLogSeverity logType, const char* file, int line, Args... args)
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
