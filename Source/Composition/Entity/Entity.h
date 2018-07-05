#pragma once

#include "Core/Utility/Types.h"
#include "EntityTraits.h"

#include <cassert>

//declare to prevent circular includes
template<typename Type>
class EntityRegistry;

template<typename Type>
class Entity
{

	friend class EntityRegistry<Type>;

	//private typesdefs
	using traits_type = EntityTraits<Type>;


public:

	//typedefs
	using entity_type = typename traits_type::entity_type;
	using version_type = typename traits_type::version_type;

	
	//construction and assignment
	~Entity();

	Entity(Entity const&) = delete;
	Entity(Entity&& o) = delete;

	Entity& operator=(Entity const&) = delete;
	Entity& operator=(Entity&& other) = delete;

protected:

	//Entities can only be created by the entity registry
	Entity();

	explicit operator Type();

private:

	Type entity_;

};

template<typename Type>
Entity<Type>::Entity() {

	//if no entities are available for reuse, create a new one
	if (EntityRegistry<Type>::availableEntities_) {

		const auto entityValue = EntityRegistry<Type>::nextAvailable;
		const auto version = EntityRegistry<Type>::entities_[entityValue] & (~traits_type::entityMask);

		entity_ = entityValue | version;
		EntityRegistry<Type>::nextAvailable = EntityRegistry<Type>::entities_[entityValue] & traits_type::entityMask;
		EntityRegistry<Type>::entities_[entityValue] = entity_;
		--EntityRegistry<Type>::availableEntities_;

	}
	else {

		entity_ = entity_type(EntityRegistry<Type>::entities_.size());
		EntityRegistry<Type>::entities_.push_back(entity_);

		//the value entityMask is the null type
		assert(entity_ < traits_type::entityMask);

	}

}

template<typename Type>
Entity<Type>::~Entity() {

	assert(valid(entity_));

	for (auto pos = EntityRegistry<Type>::pools.size(); pos; --pos) {
		auto &cpool = EntityRegistry<Type>::pools[pos - 1];

		if (cpool && cpool->has(entity_)) {
			cpool->destroy(entity_);
		}
	};

	const auto entityValue = entity_ & traits_type::entity_mask;
	const auto version = (((entity_ >> traits_type::entity_shift) + 1) & traits_type::version_mask) << traits_type::entity_shift;
	const auto node = (EntityRegistry<Type>::availableEntities_ ? EntityRegistry<Type>::nextAvailable_ : ((entityValue + 1) & traits_type::entity_mask)) | version;

	EntityRegistry<Type>::entities_[entityValue] = node;
	EntityRegistry<Type>::nextAvailable_ = entityValue;
	++EntityRegistry<Type>::availableEntities_;

}

template<typename Type>
Entity<Type>::operator Type() {
	return entity_;
}
