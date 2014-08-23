#include "Distribution.h"
#include <random>
#include <fstream>
#include <string>
#include "Utility.h"

void Distribution::loadFromFile(){
	
	std::ifstream in;
	in.open(logFileName);
	std::string str;
	while ( in ){
		getline(in, str);
		//cout <<str <<endl;
		if (str.length() < 1)
			continue;
	
		std::vector<std::string> segs = Utility::split(str, ",");
		mean = atof(segs[0].c_str());
		std = atof(segs[1].c_str());
	}
}

Distribution::Distribution(std::string _logFileName){
		logFileName = _logFileName;
		loadFromFile();
}

int Distribution::getExecutionTime(){
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean,std);

	double number = distribution(generator);
	return (int)number;
}