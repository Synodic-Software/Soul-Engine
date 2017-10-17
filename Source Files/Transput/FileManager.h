#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>


/*
	Namespace to enclose all file i/o operations.
*/
namespace FileSystem {

	/*
		Abstract base class for all file i / o classes to inherit from.
	*/
	class FileManager {
	public:

		/*
			@param		filename_		name of file to operate on
		*/
		FileManager(const std::string & filename_) :filename(filename_) {}

		/*
			@effects		read from the file specified by gilename
		*/
		virtual void Read() = 0;

		/*
			@effects		write to the file specified by filename
		*/
		virtual void Write() = 0;
	
	protected:
		std::string filename;

	};
};