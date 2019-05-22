#pragma once

#include "Structure.h"
#include "StructureConcepts.h"
#include "Span.h"


template<class T>
class ExternalStructure : public Structure, public Contiguous {

public:

	ExternalStructure() = default;
	virtual ~ExternalStructure() = default;

	virtual std::pair<nonstd::span<T>, bool> Map() = 0;
	virtual void Unmap() = 0;

};
