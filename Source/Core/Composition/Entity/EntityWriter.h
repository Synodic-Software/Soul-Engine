#pragma once

#include "Entity.h"
#include "Core/Composition/Entity/EntityStorage.h"
#include "Core/Composition/Component/Component.h"
#include "Core/Utility/ID/ClassID.h"

#include <vector>
#include <memory>
#include <cassert>


class EntityWriter: virtual public EntityStorage {

public:

	EntityWriter() = default;
	~EntityWriter() = default;

	EntityWriter(const EntityWriter&) = delete;
	EntityWriter(EntityWriter&& o) = delete;

	EntityWriter& operator=(const EntityWriter&) = delete;
	EntityWriter& operator=(EntityWriter&& other) = delete;

	// entity operations
	Entity CreateEntity();
	void RemoveEntity(Entity);


	// A component can only be attached if it is derived from the Component class
	template<typename Comp, typename... Args>
	std::enable_if_t<std::is_base_of_v<Component<Comp>, Comp>, void> AttachComponent(Entity,
		Args&&...);

	template<typename Comp>
	void RemoveComponent();


};

template<typename Comp, typename... Args>
std::enable_if_t<std::is_base_of_v<Component<Comp>, Comp>, void>
EntityWriter::AttachComponent(
	Entity entity,
	Args&&... args)
{

	assert(IsValid(entity));

	const auto componentId = ClassID::Id<Comp>();

	// componentId is always incrementing.
	if (componentId >= componentPools_.size()) {
		componentPools_.push_back(std::make_unique<SparseEntitySet<Comp>>());
	}

	auto& pool = *static_cast<SparseEntitySet<Comp>*>(componentPools_[componentId].get());

	pool.Insert(entity, std::forward<Args>(args)...);
}

template<typename Comp>
void EntityWriter::RemoveComponent()
{

	const auto componentId = ClassID::Id<Comp>();
	auto& pool = *static_cast<SparseEntitySet<Comp>*>(componentPools_[componentId].get());

	pool.Clear();
}