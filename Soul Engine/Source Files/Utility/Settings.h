#pragma once

#include <string>
#include <boost/property_tree/ptree.hpp>

namespace Settings {

	namespace detail {
		extern boost::property_tree::ptree propTree;
	}

	//Must be called from the main thread, as the setting structure is Read Only
	void Read(std::string);

	//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
	template<typename T>
	T Get(std::string propertyName, T defaultValue) {
		auto ret = detail::propTree.find(propertyName);

		if (ret == detail::propTree.not_found()) {
			detail::propTree.put(propertyName, defaultValue);
			return defaultValue;
		}

		return detail::propTree.get(propertyName, defaultValue);
	}

	//Writes the property tree to the file it opened with.
	void Write();

};