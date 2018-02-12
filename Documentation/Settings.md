# Table of Contents
1. [Description](#description)

2. [Sample Usage](#sample-usage)

3. [Public Methods](#public-methods)

4. [Private Methods](#private-methods)

5. [Enumeration Types](#enumeration-types)

# Description
A serializable hash table with strings for keys that can hold 8, 16, 32, and 64 bit integers, as well as floats and doubles as values.

# Sample Usage

## Example 1
```C++
#include "Utility/Settings.h"
#include <cstdint>

Settings::Set("a", std::int8_t(0));
std::int8_t a1 = Settings::Get("a", std::int8_t(1));
Settings::Write("b.ini",TEXT);
Settings::Read("b.ini",TEXT);
std::int8_t a2 = Settings::Get("a", std::int8_t(1));
```
Results in a1 = a2 = 0.


## Example 2
Assume "b.ini" already has "a" stored with the value 0.
```C++
#include "Utility/Settings.h"
#include <cstdint>

Settings::Read("b.ini",BINARY);
std::int8_t a = Settings::Get("a", std::int8_t(1));
```
Results in a = 0.  Would result in a = 1 if "b.ini" did not have a value for "a" already stored.  This also assumes that "b.ini" was created by a boost::archive::binary_oarchive.

# Public Methods

These specification detail how users of SoulEngine will utilize the Settings functionality.

<br>

```C++
template<typename T>
T Settings::Get(const std::string & propertyName, T defaultValue)
```
**Requires:** Template parameter T must either be explicitly given or the type of defaultValue must exactly match the stored property.  For example if T is not specified and a float is stored under "a", then defaultValue must also be a float, and not a double.

**Params:**
- const std::string & **propertyName:** The name of the property you are trying to retrieve
- T **defaultValue:** The value to return if propertyName hasn't been set already

**Effects:** None

**Returns:** If T is a [valid type](#description) and propertyName if it has already been set, returns the value associated with propertyName, and defaultValue otherwise.  

<br>

```C++
template<typename T>
bool Settings::Get(const std::string & propertyName, T defaultValue, T* propertyValue)
```
**Requires:** Template parameter T must either be explicitly given or the type of defaultValue must exactly match the stored property.  For example if T is not specified and a float is stored under "a", then defaultValue must also be a float, and not a double.

**Params:**
- const std::string & **propertyName:** The name of the property you are trying to retrieve
- T **defaultValue:** The value to return if propertyName hasn't been set already
- T * **propertyValue** The pointer to store the value of the property in; if T is a [valid type](#description) and propertyName if it has already been set, returns the value associated with propertyName, and defaultValue otherwise.  

**Effects:** Stores either the property's value or the default value in propertyValue, as described above. In addition, the default value is stored if it is not found.

**Returns:** True if T is a [valid type](#description) and the property was retrieved, or false if defaultValue was assigned to the pointer and the default value stored.

<br>

```C++
template<typename T>
T Settings::Set(const std::string & propertyName, T propertyValue)
```
**Requires:** Template parameter T must either be explicitly given or the type of defaultValue must exactly match the type you want to store the property as.  For example if T is not specified and you want to store a float 6.5 as "a", make sure 6.5 is in a float variable or given as 6.5f.

**Params:**
- const std::string & **propertyName:** The name of the property you are trying to set
- T **propertyValue:** The value to set the property as

**Effects:** Sets the value of the property, overriding any existing values.

**Returns:** True if T is a [valid type](#description) and the property was successfully set, otherwise false.

<br>

```C++
void Settings::Write(const std::string & _filename, FileType type)
```
**Requires:** type must be TEXT, XML, or BINARY.  Note binary archives are not portable.  Wide character archives are not supported.  See [boost documentation](http://www.boost.org/doc/libs/1_39_0/libs/serialization/doc/archives.html#archive_models) for more information.

**Params:**
- const std::string & **_filename:** The name of the file to serialize the settings to.
- const FileType **type** The type of boost oarchive used to serialize the file.

**Effects:** Serializes the settings in the format specified by type.  Will default to text archive if not specified.

**Returns:** None

<br>

```C++
void Settings::Read(const std::string & _filename, FileType type)
```
**Requires:** type must be TEXT, XML, or BINARY.  Note binary archives are not portable.  Wide character archives are not supported.  See [boost documentation](http://www.boost.org/doc/libs/1_39_0/libs/serialization/doc/archives.html#archive_models) for more information.

**Params:**
- const std::string & **_filename:** The name of the file to retrieve settings from.
- const FileType **type** The type of boost iarchive used to read the file.

**Effects:** Removes all current settings and retrieves the settings from the specified file.  Will default to text archive if not specified.  Note that results are undefined if the retrieval format doesn't match the serialization format.  For example, if you serialized using an XML oarchive, make sure you use a XML iarchive here.

**Returns:** None

<br>



# Private Methods

Please see the comments on private methods in [this header file](https://github.com/Behemyth/Soul-Engine/blob/master/Soul%20Engine/Source%20Files/Utility/Settings.h) for more information on private methods.  Note that these methods and associated classes are not truly private, rather they are wrapped in an internal namespace.  Nevertheless, they are only intended to be used in implementing the public methods, and are not intended for the direct use of developers using SoulEngine.

# Enumeration Types
FileType
- TEXT
- BINARY
- XML

The FileType is used for the serialization of settings.  Note that the type used to write the file must match the type to read the file (or results are undefined).  Wide character streams are not supported for serialization.
