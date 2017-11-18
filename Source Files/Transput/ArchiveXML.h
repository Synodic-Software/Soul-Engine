#pragma once

#include "ArchiveBase.h"

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/nvp.hpp>

namespace FileSystem {

	/*
		Serializes an object of type T into XML.
	*/
	template <typename T>
	class ArchiveXML : public ArchiveBase<T> {
	
	public:
		/*
			@param		filename_		specified by FileManager
			@param		object_			specified by ArchiveBase
		*/
		ArchiveXML(const std::string & filename_, T * object_) : ArchiveBase(filename_, object_) {}

		/*
			Serializes object as XML stored in the file specified by filename
		*/
		void Read() override {
			std::ifstream ifs(filename);
			boost::archive::xml_iarchive ar(ifs);
			ar & boost::serialization::make_nvp("object", *object); 
		}

		/*
		 	Loads the object serialized in XML file specified by filename into object
		 */
		void Write() override {
			std::ofstream ofs(filename);
			boost::archive::xml_oarchive ar(ofs);
			ar & boost::serialization::make_nvp("object", *object);
		}
	};
};
