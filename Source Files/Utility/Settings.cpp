#include <iostream>
#include <fstream>
#include "Settings.h"

//Will try to minimize size of boost variants.  May have a negative impact on performance.
//For more info see the boost::variant documentation
#define BOOST_VARIANT_MINIMIZE_SIZE


/*
	NOTE: See header file "Settings.h" for descriptions of all methods.
*/

//Publically Accessible Methods
namespace Settings {

	//"Under the hood" functions to handle settings.  Should not be used externally.
	namespace detail{

		std::string filename;
		TableWrapper tableWrapper;

	}

	void SetFilename(const std::string & _filename) {
		detail::filename = _filename;
	}
}