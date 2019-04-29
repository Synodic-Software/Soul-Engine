#pragma once

#include "Entity.h"
#include "Core/Composition/Component/Component.h"
#include "Core/Structure/IntrusiveSparseSet.h"

#include <vector>
#include <memory>


class EntityStorage {

public:

	EntityStorage();
	~EntityStorage() = default;

	EntityStorage(const EntityStorage&) = delete;
	EntityStorage(EntityStorage&& o) = delete;

	EntityStorage& operator=(const EntityStorage&) = delete;
	EntityStorage& operator=(EntityStorage&& other) = delete;

	bool IsValid(Entity) const noexcept;

protected:

	std::vector<std::unique_ptr<IntrusiveSparseSet<Entity>>> componentPools_;
	std::vector<Entity> entities_;

	size_t availableEntities_;
	uint64 nextAvailable_;

};