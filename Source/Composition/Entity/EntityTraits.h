#pragma once

#include "Core/Utility/Types.h"

//force specialization
template<typename>
class EntityTraits;

template<>
class EntityTraits<uint8>
{

public:

	using entity_type = uint8;
	using version_type = bool;

	static constexpr auto entityMask = 0x7F;
	static constexpr auto versionMask = 0x1;
	static constexpr auto entityBitCount = 7;

};

template<>
class EntityTraits<uint16>
{

public:

	using entity_type = uint16;
	using version_type = uint8;

	static constexpr auto entityMask = 0xFFF;
	static constexpr auto versionMask = 0xF;
	static constexpr auto entityBitCount = 12;

};

template<>
class EntityTraits<int32> {

public:

	using entity_type = int32;
	using version_type = uint16;

	static constexpr auto entityMask = 0xFFFFF;
	static constexpr auto versionMask = 0xFFF;
	static constexpr auto entityBitCount = 20;

};

template<>
class EntityTraits<uint64> {

public:

	using entity_type = uint64;
	using version_type = int32;

	static constexpr auto entityMask = 0xFFFFFFFF;
	static constexpr auto versionMask = 0xFFFFFFFF;
	static constexpr auto entityBitCount = 32;

};