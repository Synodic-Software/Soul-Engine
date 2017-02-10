#include "Settings.h"

//#define BOOST_VARIANT_USE_RELAXED_GET_BY_DEFAULT
//#define BOOST_NOEXCEPT
//#define BOOST_VARIANT_MINIMIZE_SIZE

//#include <boost/property_tree/ini_parser.hpp>
//#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>



static std::string filename;
//static Settings::TableWrapper tableWrapper;

/*
	TODO: 
	1. Hide Table and TableWrapper in an anonymous namespace.  
	2. Create static TableWrapper. - Done
	3. Create functions in settings namespace that utilize
	static TableWrapper instance (basically another wrapper)
*/

namespace Settings {

	/*Table Table::ReadTable(const std::string & filename) {
		Table table(filename);
		table.Read(filename);
		return table;
	}*/

	template <class V, typename T>
	std::pair<T, bool> Table::get(const std::unordered_map<std::string, V> & dict, 
		const std::string & propertyName, T defaultValue) {
		auto itr = dict.find(propertyName);
		if (itr == dict.end()) return std::make_pair(defaultValue,false);
		try {
			return std::make_pair(boost::relaxed_get<T>(itr->second),true);
		}
		catch (boost::bad_get e) {
			return std::make_pair(defaultValue,false);
		}

		return std::make_pair(defaultValue, false);
		
	}


	/*
		Fix: use std::enable if to create different functions, where some take the right types
		and set dict value (and return true), and others take the wrong types and return false.
		May either use multiple function signatures (propbably will have to) and/or std::enable_if.
	*/

	//WATCH OUT FOR ILLEGAL TWO IDENTICAL METHODS WITH ONLY DIFFEREING TEMPLATES
	//May have to hard code values of legal types such as first U as int8_t/uint8_t

	//enable if U is int8_t or uint8_t
	//void Table::set(OBM & dict, const std::string & propertyName, U propertyValue);
	//set dict value, return true

	//backup
	//void Table::set(OBM & dict, const std::string & propertyName, U propertyValue);
	//return false

	//template <class T, typename std::enable_if<(std::is_same<T, std::int8_t>::value || std::is_same<T, std::uint8_t>::value), int>::type = 0> 
	//bool Table::set(OBM & dict, const std::string & propertyName, T propertyValue) {
	//	dict[propertyName] = OneByteTypes(propertyValue);
	//	return true;
	//}

	//template <class T>
	//bool Table::set(OBM & dict, const std::string & propertyName, T propertyValue) { return false; }

	//template <class V, typename T>
	//void Table::set(std::unordered_map<std::string, V> & dict,
	//	const std::string & propertyName, T propertyValue) {
	//	/*V val;
	//	val = propertyValue;
	//	dict[propertyName] = val;*/
	//	dict[propertyName] = V(propertyValue);
	//}

	template <typename T>
	T Table::Get(const std::string & propertyName, T defaultValue) {
		T propertyValue;

		//bool exists = Get(propertyName, defaultvalue, &T);
		//return (exists) ? propertyvalue : defaultvalue;

		Get(propertyName, defaultValue, &propertyValue);
		return propertyValue;
	}

	template <typename T>
	bool Table::Get(const std::string & propertyName, T defaultValue, T* propertyValue) {
		std::size_t s = sizeof(T);
		std::pair<T, bool> pr;

		
		switch (s) {
		case 1: {
			//auto itr = this->obt.find(propertyName);
			//if (itr == this->obt.end()) {
			//	*propertyValue = defaultValue;
			//	return false;
			//} 

			//auto hash = typeid(defaultValue).hash_code();
			//if (hash == typeid(std::int8_t).hash_code()) {
			//	*propertyValue = boost::get<std::int8_t>(itr->second);
			//	return true;
			//}
			//else if (hash == typeid(std::uint8_t).hash_code()) {
			//	*propertyValue = boost::get<std::uint8_t>(itr->second);
			//	return true;
			//}
			//else {
			//	*propertyValue = defaultValue;
			//	return false;
			//}

			pr = this->get(this->obt, propertyName, defaultValue);
			break;
		}
		case 2: {
			pr = this->get(this->tbt, propertyName, defaultValue);
			break;
		}
		case 4: {
			pr = this->get(this->fbt, propertyName, defaultValue);
			break;
		}
		case 8: {
			pr = this->get(this->ebt, propertyName, defaultValue);
			break;
		}
		default: {
			*propertyValue = defaultValue;
			return false;
		}
		}
		*propertyValue = pr.first;
		return pr.second;
	}



	template <typename T>
	bool Table::Set(const std::string & propertyName, T propertyValue) {
		//std::size_t s = sizeof(T);
		//switch (s) {
		//case 1: {
		//	////auto hash = typeid(propertyValue).hash_code();
		//	////if (hash == typeid(std::int8_t).hash_code()) this->obt[propertyName] = OneByteTypes(propertyValue);
		//	//OneByteTypes val;
		//	//val = propertyValue;
		//	//this->obt[propertyName] = val;
		//	////this->obt[propertyName] = OneByteTypes(propertyValue);
		//	this->set(this->obt, propertyName, propertyValue);
		//	break;
		//}
		//case 2: {
		//	//static_assert(std::is_base_of<std::int16_t, T>::value || std::is_base_of<std::uint16_t, T>::value, "Invalid storage type");
		//	this->set(this->tbt, propertyName, propertyValue);
		//	break;
		//}
		////case 4: {
		////	this->set(this->fbt, propertyName, propertyValue);
		////	break;
		////}
		////case 8: {
		////	this->set(this->ebt, propertyName, propertyValue);
		////	break;
		////}
		//default: return false;
		//}
		//return true;
		if (std::is_same<std::int8_t, T>::value || std::is_same<std::uint8_t, T>::value) 
			return this->set(this->obt, propertyName, propertyValue);
		else if (std::is_same<std::int16_t, T>::value || std::is_same<std::uint16_t, T>::value) 
			return this->set(this->tbt, propertyName, propertyValue);
		else if (std::is_same<std::int32_t, T>::value || std::is_same<std::uint32_t, T>::value || std::is_same<float,T>::value)
			return this->set(this->fbt, propertyName, propertyValue);
		else if (std::is_same<std::int64_t, T>::value || std::is_same<std::uint64_t, T>::value || std::is_same<double,T>::value)
			return this->set(this->ebt, propertyName, propertyValue);
		else return false;

		//return true;
	}

	template <class Archive>
	void Table::serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::make_nvp("OneByteTypes",this->obt);
		ar & boost::serialization::make_nvp("TwoByteTypes", this->tbt);
		ar & boost::serialization::make_nvp("FourByteTypes", this->fbt);
		ar & boost::serialization::make_nvp("EightByteTypes", this->ebt);
		//ar & BOOST_SERIALIZATION_NVP(this->filename);
	}

	//void Table::Read(const std::string & filename) {
	//	std::ifstream ifs(filename);
	//	boost::archive::xml_iarchive ar(ifs);
	//	//ar & 
	//}

	//Table* Table::Read(const std::string & filename) {
	//	std::ifstream ifs(filename);
	//	boost::archive::xml_iarchive ar(ifs);
	//	Table* ptr = new Table(filename);
	//	ar & *ptr;
	//	return ptr;
	//}

	//void Table::Write() {
	//	std::ofstream ofs(filename);
	//	boost::archive::xml_oarchive ar(ofs);
	//	ar & *this;
	//}





	//namespace detail {
	//	boost::property_tree::ptree propTree;
	//}

	//void Read(std::string fn) {
	//	filename = fn;

	//	boost::filesystem::path file(filename);

	//	if (!boost::filesystem::exists(file))
	//	{
	//		std::ofstream outfile(filename);
	//		outfile.close();
	//	}


	//	boost::property_tree::ini_parser::read_ini(filename, detail::propTree);
	//}

	//void Write() {
	//	boost::property_tree::ini_parser::write_ini(filename, detail::propTree);
	//}

	void TableWrapper::Read(const std::string & _filename){
		filename = _filename;
		std::ifstream ifs(filename);
		boost::archive::text_iarchive ar(ifs);
		Table* ptr = new Table();
		ar & boost::serialization::make_nvp("Table",*ptr);
		if (this->table != nullptr) delete this->table;
		this->table = ptr;
	}

	void TableWrapper::Write() {
		std::ofstream ofs(filename);
		boost::archive::text_oarchive ar(ofs);
		ar & boost::serialization::make_nvp("Table",*(this->table));
	}

}



int main() {
	//Settings::Table st;
	//std::cout << "Initial Size: " << sizeof(st) << std::endl;
	//bool y = st.Set("a", std::int8_t(0));
	//st.Set("b", std::uint8_t(42));
	//std::int8_t val = st.Get("a", std::int8_t(1));
	//std::uint8_t val2 = st.Get("b", std::uint8_t(21));
	//std::cout << "Final Size: " << sizeof(st) << std::endl;
	//std::cout << "Values: " << static_cast<int>(val) << " " << static_cast<unsigned int>(val2) << std::endl;
	//int brkpnt = 0;
	
	filename = "a.ini";
	Settings::TableWrapper tw;
	tw.Set("a", std::int8_t(0));
	tw.Set("b", std::uint8_t(42));
	std::int8_t val = tw.Get("a", std::int8_t(1));
	std::uint8_t val2 = tw.Get("b", std::uint8_t(21));
	tw.Write();

	Settings::TableWrapper tw2;
	tw2.Read(filename);
	std::int8_t val_ = tw2.Get("a", std::int8_t(1));
	std::uint8_t val2_ = tw2.Get("b", std::uint8_t(21));

	std::cout << "Original Value of 'a': " << static_cast<int>(val) << ".  Recovered value: " << static_cast<int>(val_) << std::endl;
	std::cout << "Original Value of 'b': " << static_cast<unsigned int>(val2) << ".  Recovered value: " << static_cast<unsigned int>(val2_) << std::endl;

	int bk = 0;

	return 0;
}