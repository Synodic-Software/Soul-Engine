#include "EventManager.h"

EventManager::EventManager() :
	idCounter_(0)
{
}

void EventManager::Remove(HashString::HashType channel, HashString::HashType name, eventID id) {
	eventMap_[channel][name]->Remove(id);
}

void EventManager::Remove(HashString::HashType channel, HashString::HashType name) {
	eventMap_[channel].erase(name);
}

void EventManager::Remove(HashString::HashType channel) {
	eventMap_.erase(channel);
}