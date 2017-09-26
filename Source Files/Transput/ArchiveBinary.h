#pragma once

#include "ArchiveBase.h"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

namespace FileSystem {

	/*
		Serializes an object of type T into binary.
	*/
	template <typename T>
	class ArchiveBinary : public ArchiveBase<T> {
	
	public:
		/*
			@param		filename_		specified by FileManager
			@param		object_			specified by ArchiveBase
		*/
		ArchiveBinary(const std::string & filename_, T * object_) : ArchiveBase(filename_, object_) {}

		/*
			Serializes object as binary stored in the file specified by filename
		*/
		virtual void Read() {
			std::ifstream ifs(filename);
			boost::archive::binary_iarchive ar(ifs);
			ar & *object; 
		}

		/*
		 	Loads the object serialized in binary file specified by filename into object
		 */
		virtual void Write() {
			std::ofstream ofs(filename);
			boost::archive::binary_oarchive ar(ofs);
			ar & *object;
		}
	};
};
