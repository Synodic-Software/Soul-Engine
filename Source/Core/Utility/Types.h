#pragma once

//TODO replace with constexpr variables in metircs namespace
constexpr int KILOMETER = 100;
constexpr int HECTOMETER = 10;
constexpr int DECAMETER = 1;
constexpr float METER = 0.1f;
constexpr float DECIMETER = 0.01f;
constexpr float CENTIMETER = 0.001f;
constexpr float MILLIMETER = 0.0001f;
constexpr float DAY = 86400.0f;
constexpr float HOUR = 3600.0f;
constexpr float MINUTE = 60.0f;
constexpr float SECOND = 1.0f;
constexpr float MILLISECOND = 0.001f;
constexpr float PI = 3.141592653590f;
constexpr float TWO_PI = 6.28318530718f;
constexpr float EPSILON = 0.00000001f;
constexpr float GRAVITY = -9.80665f;
constexpr float NOON = HOUR * 12;



//global type helpers
using uint8 = unsigned char;
using int8 = signed char;

using uint16 = unsigned short;
using int16 = signed short;

using uint32 = unsigned int;
using int32 = signed int;
using uint = unsigned int;

using int64 =long long;
using uint64 = unsigned long long;


class Metrics{

};