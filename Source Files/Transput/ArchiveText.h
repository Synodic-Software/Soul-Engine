#pragma once

#include "ArchiveBase.h"

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
		void Read() override {

		}

		/*
		 @effects		loads the object serialized in text file specified by filename into object
		 */
		void Write() override {

		}
	};
};