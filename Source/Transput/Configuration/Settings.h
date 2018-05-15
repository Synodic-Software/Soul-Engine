#pragma once

//Standard Library includes
#include <string>

#include "Transput/Serialization/ArchiveBase.h"

/* Values that represent file types. */
enum FileType { null, BINARY, XML, TEXT };

//Publically Accessible Methods
namespace Settings {

	//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
	//If a template T is not explicitly provided, defaultvalue must be the same exact type as the value you 
	//are retrieving.  For example, if you are retrieving a float, default value must be 1.0f and not 1.
	//default values are not created
	

	/*
	 *    Gets.
	 *    @param	propertyName	Name of the property.
	 *    @param	defaultValue	The default value.
	 *    @return	A T.
	 */
	template<typename T>
	T Get(const std::string & propertyName, T defaultValue) {
		return defaultValue;
	}

	//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
	//If a template T is not explicitly provided, defaultvalue must be the same exact type as the value you 
	//are retrieving.  For example, if you are retrieving a float, default value must be 1.0f and not 1.
	//Retrieved value is assigned to the given pointer.
	//returns true if value was retrieved without being created, and false if defaultvalue was assigned to pointer
	//and the value was created.
	

	/*
	 *    Gets.
	 *    @param 		 	propertyName 	Name of the property.
	 *    @param 		 	defaultValue 	The default value.
	 *    @param [in,out]	propertyValue	If non-null, the property value.
	 *    @return	True if it succeeds, false if it fails.
	 */
	template<typename T>
	bool Get(const std::string & propertyName, T defaultValue, T& propertyValue) {
		propertyValue = defaultValue;
		return false;
	}

	//Set value, overriding any point
	//Returns true if data was successfully set, else false
	
	/*
	 *    Sets.
	 *    @param	propertyName 	Name of the property.
	 *    @param	propertyValue	The property value.
	 *    @return	True if it succeeds, false if it fails.
	 */

	template<typename T>
	bool Set(const std::string & propertyName, T propertyValue) {
		return false;
	}

	/*
	 *    Writes settings to the file _filename T is a boost text, XML, or binary oarchive.  Wide
	 *    character archives are not supported.
	 *    @param	_filename	Filename of the file.
	 *    @param	type	 	(Optional) The type.
	 */

	void Write(const std::string & _filename, FileType type = TEXT);

	/*
	 *    Deletes all currently stored settings and reads new ones from _filename. T is a boost
	 *    text, XML, or binary iarchive.  Wide character archives are not supported.
	 *    @param	_filename	Filename of the file.
	 *    @param	type	 	(Optional) The type.
	 */

	void Read(const std::string & _filename, FileType type = TEXT);

	/*
	 *   Deletes the underlying archive and sets the current type to null.
	 */

}