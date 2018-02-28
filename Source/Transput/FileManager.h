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
				@effects				read from the file specified by filename
		*/
		void Read(std::string& filename);
	
		/*
				@effects				write to the file specified by the filename
		*/
		void Write(std::string& filename);
	};