#include "FaceClass.h"
#include "Utility.h"
#include <string>
#include <iostream>
/**
FaceClass::FaceClass(std::string response_str){

		std::vector<std::string> faceSegs = Utility::split(response_str, ",");
		if (faceSegs.size() !=  8){
			std::cout << "Problematic string:" << response_str << " size:"<< faceSegs.size() << std::endl;
			exit (EXIT_FAILURE);
		}
		int faceID = atoi(faceSegs[0].c_str());
		int label = atoi(faceSegs[1].c_str());
		int left = atoi(faceSegs[2].c_str());
		int top = atoi(faceSegs[3].c_str());
		int width = atoi(faceSegs[4].c_str());
		int height = atoi(faceSegs[5].c_str());
		double confidence = atof(faceSegs[6].c_str());
		bool wantMore = false;
		if (atoi(faceSegs[7].c_str()) == 1){
			wantMore = true;
		}
		ewma.reserve(225);
		for (int i = 0; i < ewma.size(); ++i){
			ewma.at(i) = 1/225.0;
		}
}**/

void FaceClass::updateEWMA(std::vector<double> prediction){
	double alpha = 0.7;
	for(int i = 0; i < ewma.size(); ++i){
		ewma.at(i) = ewma.at(i) * (1-alpha) + prediction.at(i) * alpha;
	}
}