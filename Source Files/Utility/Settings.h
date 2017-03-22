#pragma once

//Boost library inclues
#include <boost/variant.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/nvp.hpp>

//Standard Library includes
#include <string>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <typeinfo>
#include <utility>
#include <type_traits>
#include <iostream>
#include <fstream>

enum FileType { BINARY, XML, TEXT };

//Publically Accessible Methods
namespace Settings {


	//"Under the hood" functions to handle settings.  Should not be used externally.
	namespace detail {

		//Type definitions for data structures used to store settings of
		//different sizes


		//If in the future even more performance is needed, you can replace this typedef
		//with something of the following format.  You will have to write a boost serialize method.

		/*struct onebytetypes {
		union { unsigned char, char } data;
		unsigned char which;
		};*/


		typedef boost::variant<std::int8_t, std::uint8_t> OneByteTypes;
		typedef boost::variant<std::int16_t, std::uint16_t> TwoByteTypes;
		typedef boost::variant<std::int32_t, std::uint32_t, float> FourByteTypes;
		typedef boost::variant<std::int64_t, std::uint64_t, double> EightByteTypes;

		typedef std::unordered_map<std::string, OneByteTypes> OBM;
		typedef std::unordered_map<std::string, TwoByteTypes> TBM;
		typedef std::unordered_map<std::string, FourByteTypes> FBM;
		typedef std::unordered_map<std::string, EightByteTypes> EBM;


		class Table {

		private:
			OBM obt;
			TBM tbt;
			FBM fbt;
			EBM ebt;

			//Used to allow boost to serialize the object
			friend class boost::serialization::access;

			//Used for serialization of class with boost::serialization
			template <class Archive>
			void serialize(Archive & ar, const unsigned int version) {
				ar & boost::serialization::make_nvp("OneByteTypes", this->obt);
				ar & boost::serialization::make_nvp("TwoByteTypes", this->tbt);
				ar & boost::serialization::make_nvp("FourByteTypes", this->fbt);
				ar & boost::serialization::make_nvp("EightByteTypes", this->ebt);
			}

			//Helper function for retrieving settings
			template <class V, typename T>
			std::pair<T, bool> get(const std::unordered_map<std::string, V> & dict,
				const std::string & propertyName, T defaultValue) {
				auto itr = dict.find(propertyName);
				if (itr == dict.end()) return std::make_pair(defaultValue, false);
				try {
					return std::make_pair(boost::relaxed_get<T>(itr->second), true);
				}
				catch (boost::bad_get e) {
					return std::make_pair(defaultValue, false);
				}

				return std::make_pair(defaultValue, false);

			}


			//Helper Functions for setting values of each type
			//These allow data to be set only when the type of the property value
			//and unordered map match


			//Setting OneByteTypes

			bool set(OBM & dict, const std::string & propertyName, std::int8_t propertyValue) {
				dict[propertyName] = OneByteTypes(propertyValue);
				return true;
			}

			bool set(OBM & dict, const std::string & propertyName, std::uint8_t propertyValue) {
				dict[propertyName] = OneByteTypes(propertyValue);
				return true;
			}

			template <class T>
			bool set(OBM & dict, const std::string & propertyName, T propertyValue) {
				return false;
			}


			//Setting TwoByteTypes

			bool set(TBM & dict, const std::string & propertyName, std::int16_t propertyValue) {
				dict[propertyName] = TwoByteTypes(propertyValue);
				return true;
			}

			bool set(TBM & dict, const std::string & propertyName, std::uint16_t propertyValue) {
				dict[propertyName] = TwoByteTypes(propertyValue);
				return true;
			}

			template <class T>
			bool set(TBM & dict, const std::string & propertyName, T propertyValue) {
				return false;
			}


			//Setting FourByteTypes

			bool set(FBM & dict, const std::string & propertyName, std::int32_t propertyValue) {
				dict[propertyName] = FourByteTypes(propertyValue);
				return true;
			}

			bool set(FBM & dict, const std::string & propertyName, std::uint32_t propertyValue) {
				dict[propertyName] = FourByteTypes(propertyValue);
				return true;
			}

			bool set(FBM & dict, const std::string & propertyName, float propertyValue) {
				dict[propertyName] = FourByteTypes(propertyValue);
				return true;
			}

			template <class T>
			bool set(FBM & dict, const std::string & propertyName, T propertyValue) {
				return false;
			}


			//Setting EightByteTypes

			bool set(EBM & dict, const std::string & propertyName, std::int64_t propertyValue) {
				dict[propertyName] = EightByteTypes(propertyValue);
				return true;
			}

			bool set(EBM & dict, const std::string & propertyName, std::uint64_t propertyValue) {
				dict[propertyName] = EightByteTypes(propertyValue);
				return true;
			}

			bool set(EBM & dict, const std::string & propertyName, double propertyValue) {
				dict[propertyName] = EightByteTypes(propertyValue);
				return true;
			}

			template <class T>
			bool set(EBM & dict, const std::string & propertyName, T propertyValue) {
				return false;
			}


		public:
			//Default Constructor
			Table() = default;

			//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
			//If a template T is not explicitly provided, defaultvalue must be the same exact type as the value you 
			//are retrieving.  For example, if you are retrieving a float, default value must be 1.0f and not 1.
			template <typename T>
			T Get(const std::string & propertyName, T defaultValue) {
				T propertyValue;
				Get(propertyName, defaultValue, &propertyValue);
				return propertyValue;
			}

			//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
			//If a template T is not explicitly provided, defaultvalue must be the same exact type as the value you 
			//are retrieving.  For example, if you are retrieving a float, default value must be 1.0f and not 1.
			//Retrieved value is assigned to the given pointer.
			//returns true if value was retrieved, and false if defaultvalue was assigned to pointer.
			template <typename T>
			bool Get(const std::string & propertyName, T defaultValue, T* propertyValue) {
				std::size_t s = sizeof(T);
				std::pair<T, bool> pr;

				switch (s) {
				case 1: {
					pr = this->get(this->obt, propertyName, defaultValue);
					break;
				} case 2: {
					pr = this->get(this->tbt, propertyName, defaultValue);
					break;
				} case 4: {
					pr = this->get(this->fbt, propertyName, defaultValue);
					break;
				} case 8: {
					pr = this->get(this->ebt, propertyName, defaultValue);
					break;
				} default: {
					*propertyValue = defaultValue;
					return false;
				}
				}
				*propertyValue = pr.first;
				return pr.second;
			}

			//Set value, overriding any point
			//Returns true if data was successfully set, else false
			template <typename T>
			bool Set(const std::string & propertyName, T propertyValue) {
				if (std::is_same<std::int8_t, T>::value || std::is_same<std::uint8_t, T>::value)
					return this->set(this->obt, propertyName, propertyValue);
				else if (std::is_same<std::int16_t, T>::value || std::is_same<std::uint16_t, T>::value)
					return this->set(this->tbt, propertyName, propertyValue);
				else if (std::is_same<std::int32_t, T>::value || std::is_same<std::uint32_t, T>::value || std::is_same<float, T>::value)
					return this->set(this->fbt, propertyName, propertyValue);
				else if (std::is_same<std::int64_t, T>::value || std::is_same<std::uint64_t, T>::value || std::is_same<double, T>::value)
					return this->set(this->ebt, propertyName, propertyValue);
				else return false;
			}



		};

		//Simple wrapper around table to allow dynamic creation of tables.
		//This makes reading tables from files easier and more efficient.
		class TableWrapper {
		private:
			Table* table;
		public:

			//Constructor and Destructor
			TableWrapper() : table(new Table()) {}
			~TableWrapper() { delete table; }

			//Read settings from file
			//T is a boost text, xml, or binary iarchive.  Wide character archives are not supported.
			template <class T>
			void Read(const std::string & _filename) {
				detail::filename = _filename;
				std::ifstream ifs(detail::filename);
				T ar(ifs);
				//boost::archive::text_iarchive ar(ifs);
				Table* ptr = new Table();
				ar & boost::serialization::make_nvp("Table", *ptr);
				if (this->table != nullptr) delete this->table;
				this->table = ptr;
			}

			//Write settings to file
			//T is a boost text, xml, or binary oarchive.  Wide character archives are not supported.
			template <class T>
			void Write() {
				std::ofstream ofs(detail::filename);
				T ar(ofs);
				//boost::archive::text_oarchive ar(ofs);
				ar & boost::serialization::make_nvp("Table", *(this->table));
			}

			//See Settings::Table::Get(const std::string&,T) for more info
			template <typename T>
			T Get(const std::string & propertyName, T defaultValue) {
				return this->table->Get(propertyName, defaultValue);
			}

			//See Settings::Table::Get(const std::string&,T,T*) for more info
			template <typename T>
			bool Get(const std::string & propertyName, T defaultValue, T* propertyValue) {
				return this->table->Get(propertyName, defaultValue, propertyValue);
			}

			//See Settings::Table::Set(const std::string&,T) for more info
			template <typename T>
			bool Set(const std::string & propertyName, T propertyValue) {
				return this->table->Set(propertyName, propertyValue);
			}

		};


		extern std::string filename;
		extern TableWrapper tableWrapper;

	}


	//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
	//If a template T is not explicitly provided, defaultvalue must be the same exact type as the value you 
	//are retrieving.  For example, if you are retrieving a float, default value must be 1.0f and not 1.
	//default values are not created
	template<typename T>
	T Get(const std::string & propertyName, T defaultValue) {
		return detail::tableWrapper.Get<T>(propertyName, defaultValue);
	}

	//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
	//If a template T is not explicitly provided, defaultvalue must be the same exact type as the value you 
	//are retrieving.  For example, if you are retrieving a float, default value must be 1.0f and not 1.
	//Retrieved value is assigned to the given pointer.
	//returns true if value was retrieved without being created, and false if defaultvalue was assigned to pointer
	//and the value was created.
	template<typename T>
	bool Get(const std::string & propertyName, T defaultValue, T* propertyValue) {
		if (!propertyValue) {
			std::cerr << "Error: a null pointer was passed as 'propertyValue'" << std::endl;
			return false;
		}

		bool found = detail::tableWrapper.Get<T>(propertyName, defaultValue, propertyValue);
		if (!found) {
			Set(propertyName, defaultValue);
		}
		return found;
	}

	//Set value, overriding any point
	//Returns true if data was successfully set, else false
	template<typename T>
	bool Set(const std::string & propertyName, T propertyValue) {
		return detail::tableWrapper.Set<T>(propertyName, propertyValue);
	}

	//Writes settings to the file _filename
	//T is a boost text, XML, or binary oarchive.  Wide character archives are not supported.
	void Write(const std::string & _filename, FileType type = TEXT);

	//Deletes all currently stored settings and reads new ones from _filename.
	//T is a boost text, XML, or binary iarchive.  Wide character archives are not supported.
	void Read(const std::string & _filename, FileType type = TEXT);
}