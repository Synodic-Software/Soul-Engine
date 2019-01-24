#pragma once

#include "Entity.h"
#include "Composition/Component/Component.h"
#include "Core/Utility/ID/ClassID.h"
#include "SparseEntitySet/AbstractSparseEntitySet.h"
#include "SparseEntitySet/SparseEntitySet.h"

#include <vector>
#include <memory>
#include <cassert>


class EntityManager
{

	friend class Entity;


public:

	//construction and assignment
	EntityManager();
	~EntityManager() = default;

	EntityManager(const EntityManager &) = delete;
	EntityManager(EntityManager&& o) = delete;

	EntityManager& operator=(const EntityManager&) = delete;
	EntityManager& operator=(EntityManager&& other) = delete;

	//entity operations
	Entity CreateEntity();
	void RemoveEntity(Entity);

	template<typename Comp>
	Comp& GetComponent(Entity) const noexcept;

	template<typename... Comp>
	std::enable_if_t< bool(sizeof...(Comp) > 1), std::tuple<Comp&...>>
		GetComponent(Entity) const noexcept;

	bool IsValid(Entity) const noexcept;

	//component operations
	
	//A component can only be attached if it is derived from the Component class
	template<typename Comp, typename ... Args>
	std::enable_if_t<std::is_base_of_v<Component<Comp>, Comp>, void>
	AttachComponent(Entity, Args&& ...);

	template<typename Comp>
	void RemoveComponent();

private:

	std::vector<std::unique_ptr<AbstractSparseEntitySet>> componentPools_;
	std::vector<Entity> entities_;

	size_t availableEntities_;
	Entity::id_type nextAvailable_;

};

template<typename Comp>
Comp& EntityManager::GetComponent(Entity entity) const noexcept {

	assert(IsValid(entity));

	const auto componentId = ClassID::Id<Comp>();
	SparseEntitySet<Comp>& pool = *static_cast<SparseEntitySet<Comp>*>(componentPools_[componentId].get());

	return pool[entity.GetId()];
}

template<typename... Comp>
std::enable_if_t< bool(sizeof...(Comp) > 1), std::tuple<Comp&...>>
EntityManager::GetComponent(Entity entity) const noexcept {

	return std::tuple<Comp&...>{
		GetComponent<Comp>(entity)...
	};

}

template<typename Comp, typename ... Args>
std::enable_if_t<std::is_base_of_v<Component<Comp>, Comp>, void>
EntityManager::AttachComponent(Entity entity, Args&& ... args) {

	assert(IsValid(entity));

	const auto componentId = ClassID::Id<Comp>();

	//componentId is always incrementing.
	if (componentId >= componentPools_.size() ) {
		componentPools_.push_back(std::make_unique<SparseEntitySet<Comp>>());
	}

	auto& pool = *static_cast<SparseEntitySet<Comp>*>(componentPools_[componentId].get());

	pool.Insert(entity, std::forward<Args>(args)...);

}

template<typename Comp>
void EntityManager::RemoveComponent() {

	const auto componentId = ClassID::Id<Comp>();
	auto& pool = *static_cast<SparseEntitySet<Comp>*>(componentPools_[componentId].get());

	pool.Clear();

}