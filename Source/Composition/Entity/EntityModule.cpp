#include "EntityModule.h"

#include "Core/Utility/Exception/Exception.h"

#include "Modules/Default/DefaultEntityBackend.h"


EntityModule::EntityModule() :
	availableEntities_(0),
	nextAvailable_(0)
{
}

Entity EntityModule::CreateEntity() {

	Entity entityID;

	//if no entities are available for reuse, create a new one
	if (availableEntities_) {

		const auto id = nextAvailable_;
		nextAvailable_ = entities_[id].GetId();
		const auto version = entities_[id].GetVersion();

		entityID = Entity(id, version);
		entities_[id] = entityID;
		--availableEntities_;

	}
	else {

		//simply add an entity. No max entity size check, who will ever go past uint32 entities? ;P
		entityID = Entity(entities_.size());
		entities_.push_back(entityID);

	}

	return entityID;

}

void EntityModule::RemoveEntity(Entity entity) {

	assert(IsValid(entity));


	throw NotImplemented();
	/*for(auto& pool : componentPools_) {
		pool->Remove(entity.GetId());
	}*/

	//grab entity data
	const auto id = availableEntities_ ? nextAvailable_ : entity.GetId() + 1;
	const auto version = entity.GetVersion() + 1;

	//set the incremented version
	entities_[id] = Entity(id, version);
	nextAvailable_ = id;
	++availableEntities_;

}

bool EntityModule::IsValid(Entity entity) const noexcept {
	const auto id = entity.GetId();
	return id < entities_.size() && entities_[id].entity_ == entity.entity_;
}

// TODO: There will only ever be one DisplayModule system per Soul application. This will need to be
// moved to the build system per platform
std::unique_ptr<EntityModule> EntityModule::CreateModule()
{

	return std::make_unique<DefaultEntityBackend>();

}
