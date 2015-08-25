#ifndef EXAMPLES_MISC_HPP
#define EXAMPLES_MISC_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <type_traits>

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, std::vector<std::vector<T> > >::type parse_csv(std::string file, char separator = ',')
{
	std::vector<std::vector<T>> result;
	std::ifstream fin(file.c_str());
	
	int row_length = -1;

	for (std::string line; getline(fin, line);) {
		auto comment = line.find_first_of('#');
		if (comment != line.npos) {
			line = line.substr(0, comment);
		}

		std::vector<T> row;
		if (row_length > 0) {
			row.reserve(row_length);
		}

		std::istringstream line_stream(line);
		std::string cell;

		while (getline(line_stream, cell, separator)) {
			if (cell.find('.') == cell.npos){
				row.push_back(atoi(cell.c_str()));
			} else {
				row.push_back(atof(cell.c_str()));
			}
		}

		if (row.size() > 0) {
			result.push_back(row);
			row_length = row.size();
		}
	}

	return result;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, std::vector<std::vector<T> > >::type parse_csv(std::string file, char separator = ',')
{
	std::vector<std::vector<T>> result;
	std::ifstream fin(file.c_str());

	int row_length = -1;

	for (std::string line; getline(fin, line);) {
		auto comment = line.find_first_of('#');
		if (comment != line.npos) {
			line = line.substr(0, comment);
		}

		std::vector<T> row;
		if (row_length > 0) {
			row.reserve(row_length);
		}

		std::istringstream line_stream(line);
		std::string cell;

		while (getline(line_stream, cell, separator)) {
			row.push_back(atoi(cell.c_str()));
		}

		if (row.size() > 0) {
			result.push_back(row);
			row_length = row.size();
		}
	}

	return result;
}

template <typename T, typename F>
T benchmark(F f) {
	auto t1 = std::chrono::steady_clock::now();

	f();

	auto t2 = std::chrono::steady_clock::now();

	return std::chrono::duration_cast<T>(t2 - t1);
}

#endif