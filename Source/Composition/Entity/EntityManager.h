#pragma once

#include "Entity.h"
#include "Composition/Component/Component.h"
#include "Core/Structures/SparseSet.h"

#include <vector>
#include <memory>

class EntityManager
{

	friend class Entity;


public:

	//construction and assignment
	EntityManager();
	~EntityManager() = default;

	EntityManager(EntityManager const&) = delete;
	EntityManager(EntityManager&& o) = delete;

	EntityManager& operator=(EntityManager const&) = delete;
	EntityManager& operator=(EntityManager&& other) = delete;

	//entity operations
	Entity CreateEntity();
	void RemoveEntity(Entity);

	bool IsValid(Entity);

	//component operations
	template<typename Comp, typename ... Args>
	void Attach(Args&& ...);


private:

	std::vector<std::unique_ptr<AbstractSparseSet>> componentPools_;
	std::vector<Entity> entities_;

	size_t availableEntities_;
	Entity::id_type nextAvailable_;

};