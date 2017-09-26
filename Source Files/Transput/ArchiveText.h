#pragma once

#include "ArchiveBase.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

namespace FileSystem {

	/*
		Serializes an object of type T into plaintext.
	*/
	template <typename T>
	class ArchiveText : public ArchiveBase<T> {
	
	public:
		/*
			@param		filename_		specified by FileManager
			@param		object_			specified by ArchiveBase
		*/
		ArchiveText(const std::string & filename_, T * object_) : ArchiveBase(filename_, object_) {}

		/*
			@effects		serializes object as text stored in the file specified by filename
		*/
		virtual void Read() {
			std::ifstream ifs(filename);
			boost::archive::text_iarchive ar(ifs);
			ar & *object; 
		}

		/*
		 @effects		loads the object serialized in text file specified by filename into object
		 */
		virtual void Write() {
			std::ofstream ofs(filename);
			boost::archive::text_oarchive ar(ofs);
			ar & *object;
		}
	};
};