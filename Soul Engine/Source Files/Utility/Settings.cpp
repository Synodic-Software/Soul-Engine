#include "Settings.h"
#include <iostream>
#include <fstream>

//Only needed for debugging
#include <cassert>

//Will try to minimize size of boost variants.  May have a negative impact on performance.
//For more info see the boost::variant documentation
#define BOOST_VARIANT_MINIMIZE_SIZE


/*
	NOTE: See header file "Settings.h" for descriptions of all methods.
*/



static std::string filename;
static internal::TableWrapper tableWrapper;


//"Under the hood" functions to handle settings.  Should not be used externally.
namespace internal{

	template <class V, typename T>
	std::pair<T, bool> Table::get(const std::unordered_map<std::string, V> & dict,
		const std::string & propertyName, T defaultValue) {
		auto itr = dict.find(propertyName);
		if (itr == dict.end()) return std::make_pair(defaultValue, false);
		try {
			return std::make_pair(boost::relaxed_get<T>(itr->second), true);
		} catch (boost::bad_get e) {
			return std::make_pair(defaultValue, false);
		}

		return std::make_pair(defaultValue, false);

	}

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

	template <typename T>
	bool Table::Set(const std::string & propertyName, T propertyValue) {
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

	template <class Archive>
	void Table::serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::make_nvp("OneByteTypes", this->obt);
		ar & boost::serialization::make_nvp("TwoByteTypes", this->tbt);
		ar & boost::serialization::make_nvp("FourByteTypes", this->fbt);
		ar & boost::serialization::make_nvp("EightByteTypes", this->ebt);
	}

	template <class T>
	void TableWrapper::Read(const std::string & _filename){
		filename = _filename;
		std::ifstream ifs(filename);
		T ar(ifs);
		//boost::archive::text_iarchive ar(ifs);
		Table* ptr = new Table();
		ar & boost::serialization::make_nvp("Table", *ptr);
		if (this->table != nullptr) delete this->table;
		this->table = ptr;
	}

	template <class T>
	void TableWrapper::Write() {
		std::ofstream ofs(filename);
		T ar(ofs);
		//boost::archive::text_oarchive ar(ofs);
		ar & boost::serialization::make_nvp("Table", *(this->table));
	}

//Tests

//Tests to make sure settings works properly when given proper inputs

	void testGetProperInputs() {
		assert(Settings::Set("a", std::int8_t(0)));
		assert(Settings::Set("b", std::uint8_t(1)));
		assert(Settings::Set("c", std::int16_t(2)));
		assert(Settings::Set("d", std::uint16_t(3)));
		assert(Settings::Set("e", std::int32_t(4)));
		assert(Settings::Set("f", std::uint32_t(5)));
		assert(Settings::Set("g", float(6.6)));
		assert(Settings::Set("h", std::int64_t(7)));
		assert(Settings::Set("i", std::uint64_t(8)));
		assert(Settings::Set("j", 9.9));

		assert(Settings::Get("a", std::int8_t(1)) == 0);
		assert(Settings::Get("b", std::uint8_t(2)) == 1);
		assert(Settings::Get("c", std::int16_t(3)) == 2);
		assert(Settings::Get("d", std::uint16_t(4)) == 3);
		assert(Settings::Get("e", std::int32_t(5)) == 4);
		assert(Settings::Get("f", std::uint32_t(6)) == 5);
		assert(Settings::Get("g", float(7.7)) == 6.6f);
		assert(Settings::Get("h", std::int64_t(8)) == 7);
		assert(Settings::Get("i", std::uint64_t(9)) == 8);
		assert(Settings::Get("j", 10.1) == 9.9);
	}

	void testGetProperInputs2() {
		assert(Settings::Set("a", std::int8_t(0)));
		assert(Settings::Set("b", std::uint8_t(1)));
		assert(Settings::Set("c", std::int16_t(2)));
		assert(Settings::Set("d", std::uint16_t(3)));
		assert(Settings::Set("e", std::int32_t(4)));
		assert(Settings::Set("f", std::uint32_t(5)));
		assert(Settings::Set("g", float(6.6)));
		assert(Settings::Set("h", std::int64_t(7)));
		assert(Settings::Set("i", std::uint64_t(8)));
		assert(Settings::Set("j", 9.9));

		std::int8_t  a; std::uint8_t  b;
		std::int16_t c; std::uint16_t d;
		std::int32_t e; std::uint32_t f; float  g;
		std::int64_t h; std::uint64_t i; double j;

		assert(Settings::Get("a", std::int8_t(1), &a));
		assert(Settings::Get("b", std::uint8_t(2), &b));
		assert(Settings::Get("c", std::int16_t(3), &c));
		assert(Settings::Get("d", std::uint16_t(4), &d));
		assert(Settings::Get("e", std::int32_t(5), &e));
		assert(Settings::Get("f", std::uint32_t(6), &f));
		assert(Settings::Get("g", float(7.7), &g));
		assert(Settings::Get("h", std::int64_t(8), &h));
		assert(Settings::Get("i", std::uint64_t(9), &i));
		assert(Settings::Get("j", 10.1, &j));

		assert(a == std::int8_t(0));
		assert(b == std::uint8_t(1));
		assert(c == std::int16_t(2));
		assert(d == std::uint16_t(3));
		assert(e == std::int32_t(4));
		assert(f == std::uint32_t(5));
		assert(g == 6.6f);
		assert(h == std::int64_t(7));
		assert(i == std::uint64_t(8));
		assert(j == 9.9);
	}

//Tests to make sure defaultValue used when an undefined 
//setting is retrived

	void testGetUndefinedSettings() {
		assert(Settings::Set("a", std::int8_t(0)));
		assert(Settings::Set("b", std::uint8_t(1)));
		assert(Settings::Set("c", std::int16_t(2)));
		assert(Settings::Set("d", std::uint16_t(3)));
		assert(Settings::Set("e", std::int32_t(4)));
		assert(Settings::Set("f", std::uint32_t(5)));
		assert(Settings::Set("g", float(6.6)));
		assert(Settings::Set("h", std::int64_t(7)));
		assert(Settings::Set("i", std::uint64_t(8)));
		assert(Settings::Set("j", 9.9));

		assert(Settings::Get("j", std::int8_t(1)) == 1);
		assert(Settings::Get("i", std::uint8_t(2)) == 2);
		assert(Settings::Get("h", std::int16_t(3)) == 3);
		assert(Settings::Get("g", std::uint16_t(4)) == 4);
		assert(Settings::Get("f", std::int32_t(5)) == 5);
		assert(Settings::Get("e", std::uint32_t(6)) == 6);
		assert(Settings::Get("d", float(7.7)) == 7.7f);
		assert(Settings::Get("c", std::int64_t(8)) == 8);
		assert(Settings::Get("b", std::uint64_t(9)) == 9);
		assert(Settings::Get("a", 10.1) == 10.1);
	}

	void testGetUndefinedSettings2() {
		assert(Settings::Set("a", std::int8_t(0)));
		assert(Settings::Set("b", std::uint8_t(1)));
		assert(Settings::Set("c", std::int16_t(2)));
		assert(Settings::Set("d", std::uint16_t(3)));
		assert(Settings::Set("e", std::int32_t(4)));
		assert(Settings::Set("f", std::uint32_t(5)));
		assert(Settings::Set("g", float(6.6)));
		assert(Settings::Set("h", std::int64_t(7)));
		assert(Settings::Set("i", std::uint64_t(8)));
		assert(Settings::Set("j", 9.9));

		std::int8_t  a; std::uint8_t  b;
		std::int16_t c; std::uint16_t d;
		std::int32_t e; std::uint32_t f; float  g;
		std::int64_t h; std::uint64_t i; double j;

		assert(!Settings::Get("j", std::int8_t(1), &a));
		assert(!Settings::Get("i", std::uint8_t(2), &b));
		assert(!Settings::Get("h", std::int16_t(3), &c));
		assert(!Settings::Get("g", std::uint16_t(4), &d));
		assert(!Settings::Get("f", std::int32_t(5), &e));
		assert(!Settings::Get("e", std::uint32_t(6), &f));
		assert(!Settings::Get("d", float(7.7), &g));
		assert(!Settings::Get("c", std::int64_t(8), &h));
		assert(!Settings::Get("b", std::uint64_t(9), &i));
		assert(!Settings::Get("a", 10.1, &j));

		assert(a == std::int8_t(1));
		assert(b == std::uint8_t(2));
		assert(c == std::int16_t(3));
		assert(d == std::uint16_t(4));
		assert(e == std::int32_t(5));
		assert(f == std::uint32_t(6));
		assert(g == 7.7f);
		assert(h == std::int64_t(8));
		assert(i == std::uint64_t(9));
		assert(j == 10.1);
	}

//Serialization Tests

	template <class T1, class T2>
	void testSerialization() {
		assert(Settings::Set("a", std::int8_t(0)));
		assert(Settings::Set("b", std::uint8_t(1)));
		assert(Settings::Set("c", std::int16_t(2)));
		assert(Settings::Set("d", std::uint16_t(3)));
		assert(Settings::Set("e", std::int32_t(4)));
		assert(Settings::Set("f", std::uint32_t(5)));
		assert(Settings::Set("g", float(6.6)));
		assert(Settings::Set("h", std::int64_t(7)));
		assert(Settings::Set("i", std::uint64_t(8)));
		assert(Settings::Set("j", 9.9));

		Settings::SetFilename("b.ini");
		Settings::Write<T2>();
		Settings::Read<T1>("b.ini");

		assert(Settings::Get("a", std::int8_t(1)) == 0);
		assert(Settings::Get("b", std::uint8_t(2)) == 1);
		assert(Settings::Get("c", std::int16_t(3)) == 2);
		assert(Settings::Get("d", std::uint16_t(4)) == 3);
		assert(Settings::Get("e", std::int32_t(5)) == 4);
		assert(Settings::Get("f", std::uint32_t(6)) == 5);
		assert(Settings::Get("g", float(7.7)) == 6.6f);
		assert(Settings::Get("h", std::int64_t(8)) == 7);
		assert(Settings::Get("i", std::uint64_t(9)) == 8);
		assert(Settings::Get("j", 10.1) == 9.9);
	}

	template <class T1, class T2>
	void testSerialization2() {
		assert(Settings::Set("a", std::int8_t(0)));
		assert(Settings::Set("b", std::uint8_t(1)));
		assert(Settings::Set("c", std::int16_t(2)));
		assert(Settings::Set("d", std::uint16_t(3)));
		assert(Settings::Set("e", std::int32_t(4)));
		assert(Settings::Set("f", std::uint32_t(5)));
		assert(Settings::Set("g", float(6.6)));
		assert(Settings::Set("h", std::int64_t(7)));
		assert(Settings::Set("i", std::uint64_t(8)));
		assert(Settings::Set("j", 9.9));

		Settings::Write<T2>("b.ini");
		Settings::Read<T1>("b.ini");

		assert(Settings::Get("a", std::int8_t(1)) == 0);
		assert(Settings::Get("b", std::uint8_t(2)) == 1);
		assert(Settings::Get("c", std::int16_t(3)) == 2);
		assert(Settings::Get("d", std::uint16_t(4)) == 3);
		assert(Settings::Get("e", std::int32_t(5)) == 4);
		assert(Settings::Get("f", std::uint32_t(6)) == 5);
		assert(Settings::Get("g", float(7.7)) == 6.6f);
		assert(Settings::Get("h", std::int64_t(8)) == 7);
		assert(Settings::Get("i", std::uint64_t(9)) == 8);
		assert(Settings::Get("j", 10.1) == 9.9);
	}

	//Runs all tests
	void runAllTests() {
		internal::testGetProperInputs();
		internal::testGetProperInputs2();
		internal::testGetUndefinedSettings();
		internal::testGetUndefinedSettings2();

		internal::testSerialization<boost::archive::binary_iarchive, boost::archive::binary_oarchive>();
		internal::testSerialization2<boost::archive::binary_iarchive, boost::archive::binary_oarchive>();

		internal::testSerialization<boost::archive::xml_iarchive, boost::archive::xml_oarchive>();
		internal::testSerialization2<boost::archive::xml_iarchive, boost::archive::xml_oarchive>();

		internal::testSerialization<boost::archive::text_iarchive, boost::archive::text_oarchive>();
		internal::testSerialization2<boost::archive::text_iarchive, boost::archive::text_oarchive>();

		std::cout << "All tests have successfully completed." << std::endl;
	}
}

//Publically Accessible Methods
namespace Settings {

	template<typename T>
	T Get(const std::string & propertyName, T defaultValue) {
		return tableWrapper.Get<T>(propertyName, defaultValue);
	}

	template<typename T>
	bool Get(const std::string & propertyName, T defaultValue, T* propertyValue) {
		if (propertyValue == nullptr) {
			std::cerr << "Error: a null pointer was passed as 'propertyValue'" << std::endl;
			exit(1); return false; //Which one should I do??
		}
		return tableWrapper.Get<T>(propertyName, defaultValue, propertyValue);
	}

	template<typename T>
	bool Set(const std::string & propertyName, T propertyValue) {
		return tableWrapper.Set<T>(propertyName, propertyValue);
	}

	template <class T>
	void Write() {
		if (filename == "") {
			std::cerr <<
				"Warning: Attempted to write settings to file with Settings::Write() without first" 
				"making a call to Settings::Read(const std::string &).  Use Settings::SetFilename(const std::string &)" 
				"or Settings::Write(const std::string &) to specify where the file should be saved" << std::endl;
			return;
		}

		tableWrapper.Write<T>();
	}

	template <class T>
	void Write(const std::string & _filename) {
		filename = _filename;
		tableWrapper.Write<T>();
	}

	template <class T>
	void Read(const std::string & _filename) {
		tableWrapper.Read<T>(_filename);
	}

	void SetFilename(const std::string & _filename) {
		filename = _filename;
	}
}



//int main() {
//	internal::runAllTests();
//	return 0;
//}