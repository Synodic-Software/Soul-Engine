#pragma once

#include "Entity.h"
#include "Core/Composition/Component/Component.h"
#include "Core/Utility/ID/ClassID.h"
#include "Core/Structure/Sparse/SparseHashMap.h"

#include <vector>
#include <memory>
#include <cassert>


class EntityRegistry{

	template<typename Comp>
	using storage_type = SparseHashMap<Entity, std::decay_t<Comp>>;

public:

	EntityRegistry();
	~EntityRegistry() = default;

	EntityRegistry(const EntityRegistry&) = delete;
	EntityRegistry(EntityRegistry&& o) = delete;

	EntityRegistry& operator=(const EntityRegistry&) = delete;
	EntityRegistry& operator=(EntityRegistry&& other) = delete;


	bool IsValid(Entity) const noexcept;

	Entity CreateEntity();
	void RemoveEntity(Entity);


	template<typename Comp, typename... Args>
	void AttachComponent(Entity,
		Args&&...);

	template<typename Comp>
	void RemoveComponent();

	template<typename Comp>
	void RemoveComponent(Entity);

	template<typename Comp>
	Comp& GetComponent(Entity) const noexcept;

	template<typename... Comp>
	std::enable_if_t<bool(sizeof...(Comp) > 1), std::tuple<Comp&...>> GetComponent(Entity) const
		noexcept;


private:

	std::vector<std::unique_ptr<SparseTable<Entity>>> componentPools_;
	std::vector<Entity> entities_;

	Entity::id_type availableEntities_;
	Entity::id_type nextAvailable_;


};

template<typename Comp>
Comp& EntityRegistry::GetComponent(Entity entity) const noexcept
{

	// TODO: C++20 Concepts
	static_assert(std::is_base_of<Component, Comp>::value,
		"The Comp parameter must be a subclass of Component");

	assert(IsValid(entity));

	const auto componentId = ClassID::Id<Comp>();
	auto& pool = *static_cast<storage_type<Comp>*> (componentPools_[componentId].get());

	return pool[entity];

}

template<typename... Comp>
std::enable_if_t<bool(sizeof...(Comp) > 1), std::tuple<Comp&...>> EntityRegistry::GetComponent(
	Entity entity) const noexcept
{

	return std::tuple<Comp&...> {GetComponent<Comp>(entity)...};

}

template<typename Comp, typename... Args>
void EntityRegistry::AttachComponent(
	Entity entity,
	Args&&... args)
{

	//TODO: C++20 Concepts
	static_assert(std::is_base_of<Component, Comp>::value,
		"The Comp parameter must be a subclass of Component");

	assert(IsValid(entity));

	const auto componentId = ClassID::Id<Comp>();

	// componentId is always incrementing.
	if (componentId >= componentPools_.size()) {
		componentPools_.push_back(std::make_unique<SparseHashMap<Entity, Comp>>());
	}

	auto& pool = *static_cast<storage_type<Comp>*>(componentPools_[componentId].get());

	pool.Emplace(entity, std::forward<Args>(args)...);

}

template<typename Comp>
void EntityRegistry::RemoveComponent()
{
	// TODO: C++20 Concepts
	static_assert(std::is_base_of<Component, Comp>::value,
		"The Comp parameter must be a subclass of Component");

	const auto componentId = ClassID::Id<Comp>();
	auto& pool = *static_cast<storage_type<Comp>*>(componentPools_[componentId].get());

	pool.Clear();

}

template<typename Comp>
void EntityRegistry::RemoveComponent(Entity entity)
{

	throw NotImplemented();

}
