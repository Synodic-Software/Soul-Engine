#pragma once

#include "Composition/Entity/EntityModule.h"


class DefaultEntityBackend : public EntityModule
{

public:

	//construction and assignment
	DefaultEntityBackend() = default;
	~DefaultEntityBackend() = default;

	DefaultEntityBackend(const DefaultEntityBackend &) = delete;
	DefaultEntityBackend(DefaultEntityBackend&&) = delete;

	DefaultEntityBackend& operator=(const DefaultEntityBackend&) = delete;
	DefaultEntityBackend& operator=(DefaultEntityBackend&&) = delete;


};