//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Include Files\Metrics.h.
//Declares the metrics class.

#pragma once

//A macro that defines kilometer.
#define KILOMETER 100
//A macro that defines hectometer.
#define HECTOMETER 10
//A macro that defines decameter.
#define DECAMETER 1
//A macro that defines meter.
#define METER 0.1f
//A macro that defines decimeter.
#define DECIMETER 0.01f
//A macro that defines centimeter.
#define CENTIMETER 0.001f
//A macro that defines millimeter.
#define MILLIMETER 0.0001f
//A macro that defines day.
#define DAY 86400.0f
//A macro that defines hour.
#define HOUR 3600.0f
//A macro that defines minute.
#define MINUTE 60.0f
//A macro that defines second.
#define SECOND 1.0f
//A macro that defines millisecond.
#define MILLISECOND 0.001f
//A macro that defines pi.
#define PI 3.141592653590f
//A macro that defines two pi.
#define TWO_PI 6.28318530718f
//A macro that defines epsilon.
#define EPSILON 0.00000001f
//A macro that defines gravity.
#define GRAVITY -9.80665f
//A macro that defines noon.
#define NOON HOUR * 12

//Defines an alias representing an 8-bit unsigned integer.
typedef unsigned char byte;
//Defines an alias representing an 8-bit unsigned integer.
typedef unsigned char uint8;
//Defines an alias representing an 8-bit signed integer.
typedef signed char int8;

//Defines an alias representing a 16-bit unsigned integer.
typedef unsigned short uint16;
//Defines an alias representing a 16-bit signed integer.
typedef signed short int16;

//Defines an alias representing a 32-bit unsigned integer.
typedef unsigned int uint32;
//Defines an alias representing a 32-bit signed integer.
typedef signed int int32;
//Defines an alias representing an unsigned integer.
typedef unsigned int uint;
//#define UINT_MAX 4294967295

//Defines an alias representing a 64-bit signed integer.
typedef long long int64;
//Defines an alias representing a 64-bit unsigned integer.
typedef unsigned long long uint64;

//Defines an alias representing the wchar.
typedef wchar_t wchar;

//global enumerations.
enum WindowType { WINDOWED, FULLSCREEN, BORDERLESS, EMPTY};


//A metrics.
class Metrics{
	public:

	private:

};