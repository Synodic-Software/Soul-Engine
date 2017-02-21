#include "Settings.h"
#include <iostream>
#include <fstream>

//Only needed for debugging
#include <cassert>

#define BOOST_VARIANT_MINIMIZE_SIZE

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
		}
		catch (boost::bad_get e) {
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

	void TableWrapper::Read(const std::string & _filename){
		filename = _filename;
		std::ifstream ifs(filename);
		boost::archive::text_iarchive ar(ifs);
		Table* ptr = new Table();
		ar & boost::serialization::make_nvp("Table", *ptr);
		if (this->table != nullptr) delete this->table;
		this->table = ptr;
	}

	void TableWrapper::Write() {
		std::ofstream ofs(filename);
		boost::archive::text_oarchive ar(ofs);
		ar & boost::serialization::make_nvp("Table", *(this->table));
	}


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

	void testGetNonDefinedSettings() {
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
		Settings::Write();
		Settings::Read("b.ini");

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
		internal::testSerialization();
		internal::testGetNonDefinedSettings();

		std::cout << "All tests have successfully completed." << std::endl;
	}

}

//Public API
namespace Settings {



	template<typename T>
	T Get(std::string propertyName, T defaultValue) {
		return tableWrapper.Get<T>(propertyName, defaultValue);
	}

	template<typename T>
	bool Set(std::string propertyName, T defaultValue) {
		return tableWrapper.Set<T>(propertyName, defaultValue);
	}

	void Write() {
		if (filename == "") {
			std::cerr <<
				"Warning: Attempted to write settings to file with Settings::Write() without first" 
				"making a call to Settings::Read(const std::string &).  Use Settings::SetFilename(std::string)" 
				"or Settings::Write(const std::string &) to specify where the file should be saved" << std::endl;
			return;
		}

		tableWrapper.Write();
	}

	void Write(const std::string & _filename) {
		filename = _filename;
		tableWrapper.Write();
	}

	void Read(const std::string & _filename) {
		tableWrapper.Read(_filename);
	}

	void SetFilename(const std::string & _filename) {
		filename = _filename;
	}
}



int main() {
	internal::runAllTests();
	return 0;
}