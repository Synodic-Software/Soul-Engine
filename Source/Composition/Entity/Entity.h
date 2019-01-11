#pragma once

#include "Core/Utility/Types.h"

class EntityManager;

template<typename T>
class SparseEntitySet;

class Entity
{

	//typedefs and constants
	friend class EntityManager;

	template<typename>
	friend class SparseEntitySet;

	using value_type = uint64;
	using id_type = uint32;
	using version_type = uint32;

	static constexpr auto entityMask = 0xFFFFFFFF;
	static constexpr auto versionMask = 0xFFFFFFFF;
	static constexpr auto entityBitCount = 32;
	static constexpr auto nullState = entityMask;

public:

	//construction and assignment
	Entity();
	~Entity() = default;

	Entity(const Entity&) = default;
	Entity(Entity&& o) = default;

	Entity& operator=(const Entity&) = default;
	Entity& operator=(Entity&& other) = default;

	//public funcs
	bool IsNull() const;


private:

	//Valid Entities can only be created by the entity registry
	explicit Entity(value_type);
	explicit Entity(id_type, version_type);

	explicit operator value_type() const;

	version_type GetVersion() const;
	id_type GetId() const;

	value_type entity_;

};