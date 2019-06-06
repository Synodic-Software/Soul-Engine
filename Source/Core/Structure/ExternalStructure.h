#pragma once

#include "Structure.h"
#include "StructureConcepts.h"
#include "Span.h"

class ExternalStructure : public Structure, public Contiguous {

public:

	ExternalStructure() = default;
	virtual ~ExternalStructure() = default;

};