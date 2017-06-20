#include "Settings.h"
#include <boost/filesystem.hpp>

//Will try to minimize size of boost variants.  May have a negative impact on performance.
//For more info see the boost::variant documentation
#define BOOST_VARIANT_MINIMIZE_SIZE


/*
NOTE: See header file "Settings.h" for descriptions of all methods.
*/

//Publically Accessible Methods
namespace Settings {

	//"Under the hood" functions to handle settings.  Should not be used externally.
	namespace detail {

		std::string filename;
		TableWrapper tableWrapper;

	}

	void Write(const std::string & _filename, FileType type) {

		detail::filename = _filename;

		if (type == TEXT) {
			detail::tableWrapper.Write<boost::archive::text_oarchive>();
		}
		else if (type == XML) {
			detail::tableWrapper.Write<boost::archive::xml_oarchive>();
		}
		else {
			detail::tableWrapper.Write<boost::archive::binary_oarchive>();
		}
	}

	void Read(const std::string & _filename, FileType type) {
		if (boost::filesystem::exists(_filename)) {

			if (type == TEXT) {
				detail::tableWrapper.Read<boost::archive::text_iarchive>(_filename);
			}
			else if (type == XML) {
				detail::tableWrapper.Read<boost::archive::xml_iarchive>(_filename);
			}
			else {
				detail::tableWrapper.Read<boost::archive::binary_iarchive>(_filename);
			}
		}
	}

}