#pragma once

#include <string>
#include <sstream>
#include <deque>
#include <mutex>
#include <iostream>

/* Values that represent log severities. */
enum SLogSeverity {
	S_TRACE,
	S_WARNING,
	S_ERROR,
	S_FATAL
};

/*
 *    A macro that defines log.
 *    @param	SEVERITY	The severity.
 *    @param	...			Variable arguments providing additional information.
 */

#define S_LOG(SEVERITY,...) Logger::detail::Log(SEVERITY,__FILE__, __LINE__, __VA_ARGS__)

/*
 *    A macro that defines log trace.
 *    @param	...	Variable arguments providing additional information.
 */

#define S_LOG_TRACE(...) Logger::detail::Log(S_TRACE,__FILE__, __LINE__, __VA_ARGS__)

/*
 *    A macro that defines log warning.
 *    @param	...	Variable arguments providing additional information.
 */

#define S_LOG_WARNING(...) Logger::detail::Log(S_WARNING,__FILE__, __LINE__, __VA_ARGS__)

/*
 *    A macro that defines log error.
 *    @param	...	Variable arguments providing additional information.
 */

#define S_LOG_ERROR(...) Logger::detail::Log(S_ERROR,__FILE__, __LINE__, __VA_ARGS__)

/*
 *    A macro that defines log fatal.
 *    @param	...	Variable arguments providing additional information.
 */

#define S_LOG_FATAL(...) Logger::detail::Log(S_FATAL,__FILE__, __LINE__, __VA_ARGS__)

/* . */
namespace Logger {

	//User: Do Not Touch
	namespace detail {

		template <typename T>

		/*
		 *    Logs a help.
		 *    @param [in,out]	o	A std::ostream to process.
		 *    @param 		 	t	A T to process.
		 */

		void LogHelp(std::ostream& o, T t)
		{
			o << t;
		}

		template<typename T, typename... Args>

		/*
		 *    Logs a help.
		 *    @param [in,out]	o   	A std::ostream to process.
		 *    @param 		 	t   	A T to process.
		 *    @param 		 	args	Variable arguments providing the arguments.
		 */

		void LogHelp(std::ostream& o, T t, Args... args)
		{
			LogHelp(o, t);
			LogHelp(o, args...);
		}

		/* The log severity strings[ 4] */
		extern std::string LogSeverityStrings[4];
		/* The log mut */
		extern std::mutex logMut;


		/* A log i. */
		typedef struct LogI {

			/* The message */
			std::string msg;
			/* The severity */
			SLogSeverity severity;
			/* The lune number */
			int luneNumber;
			/* Filename of the file */
			const char* filename;

		// .
		}LogI;

		/* The storage */
		extern std::deque<LogI> storage;

		/*
		 *    Setters for logging messages. Interfaced by macros.
		 *    @param [in,out]	oss 	The oss.
		 *    @param 		 	file	The file.
		 *    @param 		 	line	The line.
		 */

		void WriteInfo(std::ostream& oss, const char* file, int line);


		//Logs a message of the specified type
		template<typename... Args>

		/*
		 *    Logs.
		 *    @param	logType	Type of the log.
		 *    @param	file   	The file.
		 *    @param	line   	The line.
		 *    @param	args   	Variable arguments providing the arguments.
		 */

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

	/*
	 *    Get the next available string to write. Empty string if none written.
	 *    @return	A std::string.
	 */

	std::string Get();

	void WriteFile();
}