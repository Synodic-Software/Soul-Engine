#include "Settings.h"
#include "Transput/Serialization/ArchiveText.h"
#include "Transput/Serialization/ArchiveBinary.h"
#include "Transput/Serialization/ArchiveXML.h"
#include <boost/filesystem.hpp>

//Will try to minimize size of boost variants.  May have a negative impact on performance.
//For more info see the boost::variant documentation
#define BOOST_VARIANT_MINIMIZE_SIZE


/*
	NOTE: See header file "Settings.h" for descriptions of all methods.
*/

//Publically Accessible Methods
namespace Settings {

	/*
	 *    Writes.
	 *    @param	_filename	Filename of the file.
	 *    @param	type	 	The type.
	 */

	void Write(const std::string & _filename, FileType type) {

	}

	/*
	 *    Reads.
	 *    @param	_filename	Filename of the file.
	 *    @param	type	 	The type.
	 */

	void Read(const std::string & _filename, FileType type) {

	}

}