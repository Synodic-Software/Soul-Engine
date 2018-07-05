#pragma once

#include "Core/Utility/Types.h"
#include "Entity.h"

#include <vector>
#include <memory>
#include <unordered_set>

//inspired by entt https://github.com/skypjack/entt

template<typename Type = uint64>
class EntityRegistry
{

	friend class Entity<Type>;

public:

	//construction and assignment
	EntityRegistry();
	~EntityRegistry() = default;

	EntityRegistry(EntityRegistry const&) = delete;
	EntityRegistry(EntityRegistry&& o) = delete;

	EntityRegistry& operator=(EntityRegistry const&) = delete;
	EntityRegistry& operator=(EntityRegistry&& other) = delete;

	Entity<Type> Create();

protected:

	std::vector<std::unique_ptr<std::unordered_set<Type>>> componentPools;
	std::vector<Type> entities_;

	size_t availableEntities_;
	Type nextAvailable_;

};

template<typename Type>
EntityRegistry<Type>::EntityRegistry() :
	availableEntities_(0),
	nextAvailable_(0)
{
}

template<typename Type>
Entity<Type> EntityRegistry<Type>::Create() {
	return Entity<Type>();
}