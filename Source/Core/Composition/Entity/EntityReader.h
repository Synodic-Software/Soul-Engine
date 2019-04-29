#pragma once

#include "Core/Composition/Entity/EntityStorage.h"
#include "Core/Utility/ID/ClassID.h"

#include <vector>
#include <cassert>


class EntityReader: virtual public EntityStorage {

public:

	EntityReader() = default;
	~EntityReader() = default;

	EntityReader(const EntityReader&) = delete;
	EntityReader(EntityReader&& o) = delete;

	EntityReader& operator=(const EntityReader&) = delete;
	EntityReader& operator=(EntityReader&& other) = delete;


	template<typename Comp>
	Comp& GetComponent(Entity) const noexcept;

	template<typename... Comp>
	std::enable_if_t<bool(sizeof...(Comp) > 1), std::tuple<Comp&...>> GetComponent(Entity) const
		noexcept;

};

template<typename Comp>
Comp& EntityReader::GetComponent(Entity entity) const noexcept
{

	assert(IsValid(entity));

	const auto componentId = ClassID::Id<Comp>();
	SparseEntitySet<Comp>& pool =
		*static_cast<SparseEntitySet<Comp>*>(componentPools_[componentId].get());

	return pool[entity.GetId()];
}

template<typename... Comp>
std::enable_if_t<bool(sizeof...(Comp) > 1), std::tuple<Comp&...>> EntityReader::GetComponent(
	Entity entity) const noexcept
{

	return std::tuple<Comp&...> {GetComponent<Comp>(entity)...};
}