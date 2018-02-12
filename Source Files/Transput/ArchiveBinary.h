#pragma once

#include "ArchiveBase.h"

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
		void Read() override {

		}

		/*
		 	Loads the object serialized in binary file specified by filename into object
		 */
		void Write() override {

		}
	};
};
