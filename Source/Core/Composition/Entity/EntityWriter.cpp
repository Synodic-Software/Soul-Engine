#include "EntityWriter.h"


Entity EntityWriter::CreateEntity()
{

	Entity entityID;

	// if no entities are available for reuse, create a new one
	if (availableEntities_) {

		const auto id = nextAvailable_;
		nextAvailable_ = entities_[id].GetId();
		const auto version = entities_[id].GetVersion();

		entityID = Entity(id, version);
		entities_[id] = entityID;
		--availableEntities_;
	}
	else {

		// simply add an entity. No max entity size check, who will ever go past uint32 entities? ;P
		entityID = Entity(entities_.size());
		entities_.push_back(entityID);
	}

	return entityID;
}

void EntityWriter::RemoveEntity(Entity entity)
{

	assert(IsValid(entity));

	// TODO: re-enable
	// for (auto& pool : componentPools_) {
	//	pool->Remove(entity.GetId());
	//}

	// grab entity data
	const auto id = availableEntities_ ? nextAvailable_ : entity.GetId() + 1;
	const auto version = entity.GetVersion() + 1;

	// set the incremented version
	entities_[id] = Entity(id, version);
	nextAvailable_ = id;
	++availableEntities_;
}
