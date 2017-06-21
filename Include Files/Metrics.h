#pragma once

#define KILOMETER 100
#define HECTOMETER 10
#define DECAMETER 1
#define METER 0.1f
#define DECIMETER 0.01f
#define CENTIMETER 0.001f
#define MILLIMETER 0.0001f
#define DAY 86400.0f
#define HOUR 3600.0f
#define MINUTE 60.0f
#define SECOND 1.0f
#define MILLISECOND 0.001f
#define PI 3.141592653590f
#define TWO_PI 6.28318530718f
#define EPSILON 0.00000001f
#define GRAVITY -9.80665f
#define NOON HOUR * 12

typedef unsigned char byte;
typedef unsigned char uint8;
typedef signed char int8;

typedef unsigned short uint16;
typedef signed short int16;

typedef unsigned int uint32;
typedef signed int int32;
typedef unsigned int uint;
//#define UINT_MAX 4294967295

typedef long long int64;
typedef unsigned long long uint64;

typedef wchar_t wchar;

//global enumerations
enum WindowType { WINDOWED, FULLSCREEN, BORDERLESS, EMPTY};


class Metrics{
	public:

	private:

};