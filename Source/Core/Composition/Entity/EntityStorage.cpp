#include "EntityStorage.h"

EntityStorage::EntityStorage(): 
	availableEntities_(0), nextAvailable_(0)
{
}

bool EntityStorage::IsValid(Entity entity) const noexcept
{
	const auto id = entity.GetId();
	return id < entities_.size() && entities_[id].entity_ == entity.entity_;
}