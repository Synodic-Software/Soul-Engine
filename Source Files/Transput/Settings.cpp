#include "Settings.h"
#include "ArchiveText.h"
#include "ArchiveBinary.h"
#include "ArchiveXML.h"
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

		/* Filename of the file */
		std::string filename;
		/* The table */
		Table table;
		/* The current type of serialization */
		FileType curType = FileType::null;
		/* The current archive for settings */
		FileSystem::ArchiveBase<Settings::detail::Table> * curArchive = nullptr;

		/*
		 *   	Checks to see if the archive parameters have changed and creates a new archive if they have.
		 *
		 *		@param		_filename		the name of the new archive file
		 *		@param		type			the type of the new archive file
		 */
		void CheckArchive(const std::string & _filename, FileType type) {
			if (type == FileType::null) {
				std::cerr << "Error: cannot serialize to or from null filetype" << std::endl;
				return;
			}

			// create a new archive if parameters have changed
			if (detail::filename != _filename || detail::curType != type) {
				detail::filename = _filename;
				detail::curType = type;
				if (detail::curArchive != nullptr) delete detail::curArchive;
				switch (type) {
					case FileType::TEXT: {
						detail::curArchive = new FileSystem::ArchiveText<Settings::detail::Table>(_filename, &detail::table);
						break;
					} case FileType::BINARY: {
						detail::curArchive = new FileSystem::ArchiveBinary<Settings::detail::Table>(_filename, &detail::table);
						break;
					} case FileType::XML: {
						detail::curArchive = new FileSystem::ArchiveXML<Settings::detail::Table>(_filename, &detail::table);
						break;
					}  default: {
						std::cerr << "Error: unknown FileType " << type << std::endl;
						detail::curArchive = nullptr;
						break;
					}
				}
			}
		}

	}

	/*
	*   Deletes the underlying archive and sets the current type to null.
	*/

	void DeleteArchive() {
		if (detail::curArchive != nullptr) {
			delete detail::curArchive;
			detail::curArchive = nullptr;
		}
		detail::curType = FileType::null;
	}

	/*
	 *    Writes.
	 *    @param	_filename	Filename of the file.
	 *    @param	type	 	The type.
	 */

	void Write(const std::string & _filename, FileType type) {
		detail::CheckArchive(_filename, type);
		detail::curArchive->Write();
	}

	/*
	 *    Reads.
	 *    @param	_filename	Filename of the file.
	 *    @param	type	 	The type.
	 */

	void Read(const std::string & _filename, FileType type) {
		if (boost::filesystem::exists(_filename)) {
			detail::CheckArchive(_filename, type);
			detail::curArchive->Read();
		}
		else {
			std::cerr << "Error:: file \"" << _filename << "\" does not exist" << std::endl;
		}
	}

}