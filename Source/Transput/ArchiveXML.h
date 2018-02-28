#pragma once

#include "ArchiveBase.h"

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

		}

		/*
		 	Loads the object serialized in XML file specified by filename into object
		 */
		void Write() override {

		}
	};
};
