//Custom Assert Function
/*

FLAG for disable debugging is #define NDEBUG
*/

#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include "Logger.h"


#define S_ASSERT_EQUALS(V1,V2,...) Assert::equals<>(V1,V2,__FILE__,__LINE__)
#define S_ASSERT_NOT_EQUALS(V1,V2,...) Assert::notEquals<>(V1,V2,__FILE__,__LINE__)
#define S_ASSERT_LESS_THAN(V1,V2,...) Assert::lessThan(V1,V2,__FILE__,__LINE__)
#define S_ASSERT_GREATER_THAN(V1,V2,...) Assert::greaterThan(V1,V2,__FILE__,__LINE__)
namespace Assert
{
	template <typename T> void equals(T v1, T v2, const char* file, int line)
	{
#ifndef NDEBUG
		if (v1 == v2)
		{
			return;
		}
		else
		{
			std::stringstream ss;
			ss << "ASSERT FAILURE in file " << file << ", line " << line << ": " << v1 << " != " << v2 << ".\n";
			S_LOG_FATAL(ss.str());
			exit(1);
		}
#endif

	}
	template <typename T> void notEquals(T v1, T v2, const char* file, int line)
	{
#ifndef NDEBUG
		if (v1 != v2)
		{
			return;
		}
		else
		{
			std::stringstream ss;
			ss << "ASSERT FAILURE in file " << file << ", line " << line << ": " << v1 << " = " << v2 << ".\n";
			S_LOG_FATAL(ss.str());
			exit(1);
		}
#endif

	}
	template <typename T> void lessThan(T v1, T v2, const char* file, int line)
	{
#ifndef NDEBUG
		if (v1<v2)
		{
			return;
		}
		else
		{
			std::stringstream ss;
			ss << "ASSERT FAILURE in file " << file << ", line " << line << ": " << v1 << " !< " << v2 << ".\n";
			S_LOG_FATAL(ss.str());
			exit(1);
		}
#endif
	}

	template <typename T> void greaterThan(T v1, T v2, const char* file, int line)
	{
#ifndef NDEBUG
		if (v1>v2)
		{
			return;
		}
		else
		{
			std::stringstream ss;
			ss << "ASSERT FAILURE in file " << file << ", line " << line << ": " << v1 << " !> " << v2 << ".\n";
			S_LOG_FATAL(ss.str());
			exit(1);
		}
#endif
	}


}