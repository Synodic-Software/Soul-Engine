/*
FLAG for disable debugging is #define NDEBUG
*/

#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include "Logger.h"

//---------------------------------------------------------------------------------------------------
//A macro that defines assert equals.
//@param	V1 	The first void.
//@param	V2 	The second void.
//@param	...	Variable arguments providing additional information.

#define S_ASSERT_EQUALS(V1,V2,...) Assert::equals<>(V1,V2,__FILE__,__LINE__)

//---------------------------------------------------------------------------------------------------
//A macro that defines assert not equals.
//@param	V1 	The first void.
//@param	V2 	The second void.
//@param	...	Variable arguments providing additional information.

#define S_ASSERT_NOT_EQUALS(V1,V2,...) Assert::notEquals<>(V1,V2,__FILE__,__LINE__)

//---------------------------------------------------------------------------------------------------
//A macro that defines assert less than.
//@param	V1 	The first void.
//@param	V2 	The second void.
//@param	...	Variable arguments providing additional information.

#define S_ASSERT_LESS_THAN(V1,V2,...) Assert::lessThan(V1,V2,__FILE__,__LINE__)

//---------------------------------------------------------------------------------------------------
//A macro that defines assert greater than.
//@param	V1 	The first void.
//@param	V2 	The second void.
//@param	...	Variable arguments providing additional information.

#define S_ASSERT_GREATER_THAN(V1,V2,...) Assert::greaterThan(V1,V2,__FILE__,__LINE__)
//.
namespace Assert
{
	//---------------------------------------------------------------------------------------------------
	//Tests if objects are considered equal.
	//@tparam	T	Generic type parameter.
	//@param	v1  	The first T.
	//@param	v2  	The second T.
	//@param	file	The file.
	//@param	line	The line.

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

	//---------------------------------------------------------------------------------------------------
	//Not equals.
	//@tparam	T	Generic type parameter.
	//@param	v1  	The first T.
	//@param	v2  	The second T.
	//@param	file	The file.
	//@param	line	The line.

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

	//---------------------------------------------------------------------------------------------------
	//Less than.
	//@tparam	T	Generic type parameter.
	//@param	v1  	The first T.
	//@param	v2  	The second T.
	//@param	file	The file.
	//@param	line	The line.

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

	//---------------------------------------------------------------------------------------------------
	//Greater than.
	//@tparam	T	Generic type parameter.
	//@param	v1  	The first T.
	//@param	v2  	The second T.
	//@param	file	The file.
	//@param	line	The line.

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