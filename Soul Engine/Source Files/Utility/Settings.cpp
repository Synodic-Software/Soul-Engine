#include "Settings.h"

#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>

namespace Settings {
	namespace detail {

		std::string filename;
		boost::property_tree::ptree propTree;

	}

	void Read(std::string fn) {
		detail::filename = fn;

		boost::filesystem::path file(detail::filename);

		if (!boost::filesystem::exists(file))
		{
			std::ofstream outfile(detail::filename);
			outfile.close();
		}


		boost::property_tree::ini_parser::read_ini(detail::filename, detail::propTree);
	}

	void Write() {
		boost::property_tree::ini_parser::write_ini(detail::filename, detail::propTree);
	}

}