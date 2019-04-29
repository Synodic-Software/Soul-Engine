#pragma once

#include "Entity.h"
#include "Core/Composition/Component/Component.h"
#include "Core/Utility/ID/ClassID.h"
#include "Core/Structure/IntrusiveSparseSet.h"

#include "Core/Composition/Entity/EntityWriter.h"
#include "Core/Composition/Entity/EntityReader.h"
#include "Core/Composition/Entity/EntityStorage.h"

#include <vector>
#include <memory>
#include <cassert>


class EntityRegistry: public EntityWriter, public EntityReader{

public:

	EntityRegistry() = default;
	~EntityRegistry() = default;

	EntityRegistry(const EntityRegistry&) = delete;
	EntityRegistry(EntityRegistry&& o) = delete;

	EntityRegistry& operator=(const EntityRegistry&) = delete;
	EntityRegistry& operator=(EntityRegistry&& other) = delete;


};