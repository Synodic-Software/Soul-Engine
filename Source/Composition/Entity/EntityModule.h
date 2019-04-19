#pragma once

#include "Entity.h"
#include "Composition/Component/Component.h"
#include "Core/Utility/ID/ClassID.h"

#include <vector>
#include <memory>
#include <cassert>


class EntityModule
{


	friend class Entity;


public:

	//construction and assignment
	EntityModule();
	~EntityModule() = default;

	EntityModule(const EntityModule &) = delete;
	EntityModule(EntityModule&&) = delete;

	EntityModule& operator=(const EntityModule&) = delete;
	EntityModule& operator=(EntityModule&&) = delete;

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


	// Factory
	static std::unique_ptr<EntityModule> CreateModule();


private:

	std::vector<Entity> entities_;

	size_t availableEntities_;
	Entity::id_type nextAvailable_;

};

template<typename Comp>
Comp& EntityModule::GetComponent(Entity entity) const noexcept {

	assert(IsValid(entity));

	const auto componentId = ClassID::Id<Comp>();
	SparseEntitySet<Comp>& pool = *static_cast<SparseEntitySet<Comp>*>(componentPools_[componentId].get());

	return pool[entity.GetId()];
}

template<typename... Comp>
std::enable_if_t< bool(sizeof...(Comp) > 1), std::tuple<Comp&...>>
EntityModule::GetComponent(Entity entity) const noexcept {

	return std::tuple<Comp&...>{
		GetComponent<Comp>(entity)...
	};

}

template<typename Comp, typename ... Args>
std::enable_if_t<std::is_base_of_v<Component<Comp>, Comp>, void>
EntityModule::AttachComponent(Entity entity, Args&& ... args) {

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
void EntityModule::RemoveComponent() {

	const auto componentId = ClassID::Id<Comp>();
	auto& pool = *static_cast<SparseEntitySet<Comp>*>(componentPools_[componentId].get());

	pool.Clear();

}