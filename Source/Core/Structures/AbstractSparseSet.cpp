#include "AbstractSparseSet.h"

bool AbstractSparseSet::Find(size_t i) {
	return i < sparse_.size() && sparse_[i] != emptyValue;
}
