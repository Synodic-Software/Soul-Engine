#pragma once

#include "FileManager.h"


namespace FileSystem{

	/*
		Abstract base class for all serialization classes to inherit from.
		@param		T		type of the object to be serialized

		Note classes which inherit from AbstractBase are not required to deallocate memory for an existing object when loading a new one.
	*/
	template <typename T>
	class ArchiveBase : public FileManager {
	
	public:
		/*
			@param		filename_		specified by FileManager
			@param		object_			the object to be serialized or used to contain the loaded instance
		*/
		ArchiveBase(const std::string & filename_, T * object_) : FileManager(filename_), object(object_) {}

	protected:
		T * object;
	};
};