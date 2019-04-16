#include "DefaultEventBackend.h"

DefaultEventBackend::DefaultEventBackend() :
	idCounter_(0)
{
}

void DefaultEventBackend::Remove(HashString::HashType channel, HashString::HashType name, uint64 id)
{
	eventMap_[channel][name]->Remove(id);
}

void DefaultEventBackend::Remove(HashString::HashType channel, HashString::HashType name) {
	eventMap_[channel].erase(name);
}

void DefaultEventBackend::Remove(HashString::HashType channel) {
	eventMap_.erase(channel);
}