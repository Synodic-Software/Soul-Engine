#pragma once

//#include <boost/property_tree/ptree.hpp>
#include <boost/variant.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/static_assert.hpp>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <typeinfo>
#include <utility>
#include <type_traits>

namespace Settings {
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
		//std::string filename;
		
		//static struct : boost::static_visitor<> {
		//	bool operator(std::unordered_map<std::string,OneByteTypes> & dict)
		//} setVisitor;

		//Filename Constructor
		//Table(const std::string & filename) : obt(), tbt(), fbt(), ebt(), filename(filename) {}

		//Used to allow boost to serialize the object
		friend class boost::serialization::access;

		//Used for serialization of class with boost::serialization
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version);

		//Helper function for retrieving settings
		template <class V, typename T>
		std::pair<T, bool> get(const std::unordered_map<std::string, V> & dict,
			const std::string & propertyName, T defaultValue);

		////Helper function for adding new settings
		//template <class V, typename T>
		//void set(std::unordered_map<std::string, V> & dict,
		//	const std::string & propertyName, T propertyValue);

		//template <class T, typename std::enable_if<(std::is_same<T, std::int8_t>::value || std::is_same<T, std::uint8_t>::value), int>::type = 0>
		//bool set(OBM & dict, const std::string & propertyName, T propertyValue);

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

		bool set(FBM & dict, const std::string & propertyName, std::int32_t propertyValue) {
			dict[propertyName] = FourByteTypes(propertyValue);
			return true;
		}

		bool set(FBM & dict, const std::string & propertyName, std::uint32_t propertyValue) {
			dict[propertyName] = FourByteTypes(propertyValue);
			return true;
		}

		template <class T>
		bool set(FBM & dict, const std::string & propertyName, T propertyValue) {
			return false;
		}

		bool set(EBM & dict, const std::string & propertyName, std::int64_t propertyValue) {
			dict[propertyName] = EightByteTypes(propertyValue);
			return true;
		}

		bool set(EBM & dict, const std::string & propertyName, std::uint64_t propertyValue) {
			dict[propertyName] = EightByteTypes(propertyValue);
			return true;
		}

		template <class T>
		bool set(EBM & dict, const std::string & propertyName, T propertyValue) {
			return false;
		}


	public:
		//Default Constructor
		//Table() : obt(), tbt(), fbt(), ebt(), filename() {}
		Table() = default;

		//Read file static constructor
		//static Table ReadTable(const std::string & filename);

		//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
		//If a template T is not explicitly provided, defaultvalue must be the same exact type as the value you 
		//are retrieving.  For example, if you are retrieving a float, default value must be 1.0f and not 1.
		template <typename T>
		T Get(const std::string & propertyName, T defaultValue);

		//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
		//If a template T is not explicitly provided, defaultvalue must be the same exact type as the value you 
		//are retrieving.  For example, if you are retrieving a float, default value must be 1.0f and not 1.
		//Retrieved value is assigned to the given pointer.
		//returns true if value was retrieved, and false if defaultvalue was assigned to pointer.
		template <typename T>
		bool Get(const std::string & propertyName, T defaultValue, T* propertyValue);

		//Set value, overriding any point
		//Returns true if data was successfully set, else false
		template <typename T>
		bool Set(const std::string & propertyName, T propertyValue);

		//Must be called from the main thread, as the setting structure is Read Only
		//void Read(const std::string & filename);

		//Table* Read(const std::string & filename);

		//Writes the property tree to the file it opened with.
		//void Write();



	};

	//Simple wrapper around table to allow dynamic creation of tables.
	//This makes reading tables from files easier and more efficient.
	class TableWrapper {
	private:
		Table* table;
	public:
		TableWrapper() : table(new Table()) {}
		~TableWrapper() { delete table; }
		void Read(const std::string & _filename);
		void Write();
		
		//See header for Settings::Table::Get(const std::string&,T) for more info
		template <typename T>
		T Get(const std::string & propertyName, T defaultValue) { 
			return this->table->Get(propertyName, defaultValue); }

		//See header for Settings::Table::Get(const std::string&,T,T*) for more info
		template <typename T>
		bool Get(const std::string & propertyName, T defaultValue, T* propertyValue) { 
			return this->table->Get(propertyName, defaultValue, propertyValue); }

		//See header for Settings::Table::Set(const std::string&,T) for more info
		template <typename T>
		bool Set(const std::string & propertyName, T propertyValue) { 
			return this->table->Set(propertyName, propertyValue); }

	};

	//namespace detail {
	//	extern boost::property_tree::ptree propTree;
	//}

	//Must be called from the main thread, as the setting structure is Read Only
	//void Read(std::string);

	//Retrieves a value at the specified placeholder. Cannot accept user defined types, only system types
	//template<typename T>
	//T Get(std::string propertyName, T defaultValue) {
	//	auto ret = detail::propTree.find(propertyName);

	//	if (ret == detail::propTree.not_found()) {
	//		detail::propTree.put(propertyName, defaultValue);
	//		return defaultValue;
	//	}

	//	return detail::propTree.get(propertyName, defaultValue);
	//}

	//set value, overriding any point
	//template<typename T>
	//void Set(std::string propertyName, T defaultValue) {
	//	detail::propTree.put(propertyName, defaultValue);
	//}

	//Writes the property tree to the file it opened with.
	//void Write();

};