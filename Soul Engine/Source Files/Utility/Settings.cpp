#include "Settings.h"

#define BOOST_VARIANT_MINIMIZE_SIZE

#include <iostream>
#include <fstream>

//Only needed for debugging
#include <cassert>



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

	template <typename T>
	T Table::Get(const std::string & propertyName, T defaultValue) {
		T propertyValue;
		Get(propertyName, defaultValue, &propertyValue);
		return propertyValue;
	}

	template <typename T>
	bool Table::Get(const std::string & propertyName, T defaultValue, T* propertyValue) {
		std::size_t s = sizeof(T);
		std::pair<T, bool> pr;

		switch (s) {
		case 1: {
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
		if (std::is_same<std::int8_t, T>::value || std::is_same<std::uint8_t, T>::value) 
			return this->set(this->obt, propertyName, propertyValue);
		else if (std::is_same<std::int16_t, T>::value || std::is_same<std::uint16_t, T>::value) 
			return this->set(this->tbt, propertyName, propertyValue);
		else if (std::is_same<std::int32_t, T>::value || std::is_same<std::uint32_t, T>::value || std::is_same<float,T>::value)
			return this->set(this->fbt, propertyName, propertyValue);
		else if (std::is_same<std::int64_t, T>::value || std::is_same<std::uint64_t, T>::value || std::is_same<double,T>::value)
			return this->set(this->ebt, propertyName, propertyValue);
		else return false;
	}

	template <class Archive>
	void Table::serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::make_nvp("OneByteTypes",this->obt);
		ar & boost::serialization::make_nvp("TwoByteTypes", this->tbt);
		ar & boost::serialization::make_nvp("FourByteTypes", this->fbt);
		ar & boost::serialization::make_nvp("EightByteTypes", this->ebt);
	}

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

void testGetProperInputs() {
	Settings::TableWrapper tw;
	assert(tw.Set("a", std::int8_t(0)));
	assert(tw.Set("b", std::uint8_t(1)));
	assert(tw.Set("c", std::int16_t(2)));
	assert(tw.Set("d", std::uint16_t(3)));
	assert(tw.Set("e", std::int32_t(4)));
	assert(tw.Set("f", std::uint32_t(5)));
	assert(tw.Set("g", float(6.6)));
	assert(tw.Set("h", std::int64_t(7)));
	assert(tw.Set("i", std::uint64_t(8)));
	assert(tw.Set("j", 9.9));

	assert(tw.Get("a", std::int8_t(1)) == 0);
	assert(tw.Get("b", std::uint8_t(2)) == 1);
	assert(tw.Get("c", std::int16_t(3)) == 2);
	assert(tw.Get("d", std::uint16_t(4)) == 3);
	assert(tw.Get("e", std::int32_t(5)) == 4);
	assert(tw.Get("f", std::uint32_t(6)) == 5);
	assert(tw.Get("g", float(7.7)) == 6.6f);
	assert(tw.Get("h", std::int64_t(8)) == 7);
	assert(tw.Get("i", std::uint64_t(9)) == 8);
	assert(tw.Get("j", 10.1) == 9.9);

}

void testGetNonDefinedSettings() {
	Settings::TableWrapper tw;
	assert(tw.Set("a", std::int8_t(0)));
	assert(tw.Set("b", std::uint8_t(1)));
	assert(tw.Set("c", std::int16_t(2)));
	assert(tw.Set("d", std::uint16_t(3)));
	assert(tw.Set("e", std::int32_t(4)));
	assert(tw.Set("f", std::uint32_t(5)));
	assert(tw.Set("g", float(6.6)));
	assert(tw.Set("h", std::int64_t(7)));
	assert(tw.Set("i", std::uint64_t(8)));
	assert(tw.Set("j", 9.9));

	assert(tw.Get("j", std::int8_t(1)) == 1);
	assert(tw.Get("i", std::uint8_t(2)) == 2);
	assert(tw.Get("h", std::int16_t(3)) == 3);
	assert(tw.Get("g", std::uint16_t(4)) == 4);
	assert(tw.Get("f", std::int32_t(5)) == 5);
	assert(tw.Get("e", std::uint32_t(6)) == 6);
	assert(tw.Get("d", float(7.7)) == 7.7f);
	assert(tw.Get("c", std::int64_t(8)) == 8);
	assert(tw.Get("b", std::uint64_t(9)) == 9);
	assert(tw.Get("a", 10.1) == 10.1);

}

void testSerialization() {
	Settings::TableWrapper tw;
	assert(tw.Set("a", std::int8_t(0)));
	assert(tw.Set("b", std::uint8_t(1)));
	assert(tw.Set("c", std::int16_t(2)));
	assert(tw.Set("d", std::uint16_t(3)));
	assert(tw.Set("e", std::int32_t(4)));
	assert(tw.Set("f", std::uint32_t(5)));
	assert(tw.Set("g", float(6.6)));
	assert(tw.Set("h", std::int64_t(7)));
	assert(tw.Set("i", std::uint64_t(8)));
	assert(tw.Set("j", 9.9));

	filename = "b.ini";
	tw.Write();
	Settings::TableWrapper tw2;
	tw2.Read("b.ini");

	assert(tw2.Get("a", std::int8_t(1)) == 0);
	assert(tw2.Get("b", std::uint8_t(2)) == 1);
	assert(tw2.Get("c", std::int16_t(3)) == 2);
	assert(tw2.Get("d", std::uint16_t(4)) == 3);
	assert(tw2.Get("e", std::int32_t(5)) == 4);
	assert(tw2.Get("f", std::uint32_t(6)) == 5);
	assert(tw2.Get("g", float(7.7)) == 6.6f);
	assert(tw2.Get("h", std::int64_t(8)) == 7);
	assert(tw2.Get("i", std::uint64_t(9)) == 8);
	assert(tw2.Get("j", 10.1) == 9.9);

}

void runAllTests() {
	testGetProperInputs();
	testSerialization();
	testGetNonDefinedSettings();
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
	
	//filename = "a.ini";
	//Settings::TableWrapper tw;
	//tw.Set("a", std::int8_t(0));
	//tw.Set("b", std::uint8_t(42));
	//std::int8_t val = tw.Get("a", std::int8_t(1));
	//std::uint8_t val2 = tw.Get("b", std::uint8_t(21));
	//tw.Write();

	//Settings::TableWrapper tw2;
	//tw2.Read(filename);
	//std::int8_t val_ = tw2.Get("a", std::int8_t(1));
	//std::uint8_t val2_ = tw2.Get("b", std::uint8_t(21));

	//std::cout << "Original Value of 'a': " << static_cast<int>(val) << ".  Recovered value: " << static_cast<int>(val_) << std::endl;
	//std::cout << "Original Value of 'b': " << static_cast<unsigned int>(val2) << ".  Recovered value: " << static_cast<unsigned int>(val2_) << std::endl;

	runAllTests();

	int bk = 0;

	return 0;
}