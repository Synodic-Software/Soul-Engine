#include <iostream>
#include <fstream>

//This file is included by Settings.h to deal with templating

//Will try to minimize size of boost variants.  May have a negative impact on performance.
//For more info see the boost::variant documentation
#define BOOST_VARIANT_MINIMIZE_SIZE


/*
	NOTE: See header file "Settings.h" for descriptions of all methods.
*/


//"Under the hood" functions to handle settings.  Should not be used externally.
namespace internal{
	std::string filename;
	internal::TableWrapper tableWrapper;

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

	template <class T>
	void TableWrapper::Read(const std::string & _filename){
		internal::filename = _filename;
		std::ifstream ifs(internal::filename);
		T ar(ifs);
		//boost::archive::text_iarchive ar(ifs);
		Table* ptr = new Table();
		ar & boost::serialization::make_nvp("Table", *ptr);
		if (this->table != nullptr) delete this->table;
		this->table = ptr;
	}

	template <class T>
	void TableWrapper::Write() {
		std::ofstream ofs(internal::filename);
		T ar(ofs);
		//boost::archive::text_oarchive ar(ofs);
		ar & boost::serialization::make_nvp("Table", *(this->table));
	}

}

//Publically Accessible Methods
namespace Settings {

	template<typename T>
	T Get(const std::string & propertyName, T defaultValue) {
		return internal::tableWrapper.Get<T>(propertyName, defaultValue);
	}

	template<typename T>
	bool Get(const std::string & propertyName, T defaultValue, T* propertyValue) {
		if (propertyValue == nullptr) {
			std::cerr << "Error: a null pointer was passed as 'propertyValue'" << std::endl;
			exit(1); return false; //Which one should I do??
		}
		return internal::tableWrapper.Get<T>(propertyName, defaultValue, propertyValue);
	}

	template<typename T>
	bool Set(const std::string & propertyName, T propertyValue) {
		return internal::tableWrapper.Set<T>(propertyName, propertyValue);
	}

	template <class T>
	void Write() {
		if (internal::filename == "") {
			std::cerr <<
				"Warning: Attempted to write settings to file with Settings::Write() without first" 
				"making a call to Settings::Read(const std::string &).  Use Settings::SetFilename(const std::string &)" 
				"or Settings::Write(const std::string &) to specify where the file should be saved" << std::endl;
			return;
		}

		internal::tableWrapper.Write<T>();
	}

	template <class T>
	void Write(const std::string & _filename) {
		internal::filename = _filename;
		internal::tableWrapper.Write<T>();
	}

	template <class T>
	void Read(const std::string & _filename) {
		internal::tableWrapper.Read<T>(_filename);
	}

	void SetFilename(const std::string & _filename) {
		internal::filename = _filename;
	}
}



//int main() {
//	internal::runAllTests();
//	return 0;
//}