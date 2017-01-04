#include "Settings.h"

#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>

namespace {
	std::string filename;
}

namespace Settings {
	namespace detail {
		boost::property_tree::ptree propTree;
	}

	void Read(std::string fn) {
		filename = fn;

		boost::filesystem::path file(filename);

		if (!boost::filesystem::exists(file))
		{
			std::ofstream outfile(filename);
			outfile.close();
		}


		boost::property_tree::ini_parser::read_ini(filename, detail::propTree);
	}

	void Write() {
		boost::property_tree::ini_parser::write_ini(filename, detail::propTree);
	}

}