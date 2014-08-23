#ifndef DISTRIBUTIONCLASS_H
#define DISTRIBUTIONCLASS_H
#include <string>

class Distribution{
	public:
		std::string distributionName;
		std::string logFileName;
		double mean;
		double std;

	
	int Distribution::getExecutionTime();
	void Distribution::loadFromFile();
	Distribution::Distribution(std::string _logFileName);
};


#endif
