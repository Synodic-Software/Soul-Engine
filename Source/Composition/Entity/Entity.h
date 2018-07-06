#pragma once

#include "Core/Utility/Types.h"

//declare to prevent circular includes
class EntityManager;

class Entity
{

	//typedefs and constants
	friend class EntityManager;

	using value_type = uint64;
	using id_type = uint32;
	using version_type = uint32;

	static constexpr auto entityMask = 0xFFFFFFFF;
	static constexpr auto versionMask = 0xFFFFFFFF;
	static constexpr auto entityBitCount = 32;
	static constexpr auto null = entityMask;

public:

	//construction and assignment
	~Entity() = default;

	Entity(Entity const&) = default;
	Entity(Entity&& o) = default;

	Entity& operator=(Entity const&) = default;
	Entity& operator=(Entity&& other) = default;


private:

	//Entities can only be created by the entity registry
	Entity() = default;
	explicit Entity(value_type);
	explicit Entity(id_type, version_type);

	explicit operator value_type() const;

	version_type GetVersion() const;
	id_type GetId() const;

	value_type entity_;

};