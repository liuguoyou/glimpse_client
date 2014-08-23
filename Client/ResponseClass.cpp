#include "ResponseClass.h"
#include "Utility.h"
#include <string>
#include <iostream>


EntireFrameResponse::EntireFrameResponse(std::string response_str, double totalProcessingTime){

	//std::cout << "Parse string.." << response_str << std::endl;
	std::vector<std::string> segs;
	segs = Utility::split(response_str, ":");
	if (segs.size() == 0){
		std::cout << "Problematic string:" << response_str << std::endl;
		exit (EXIT_FAILURE);
	}
	faceNumber = atoi(segs[0].c_str());
	if (faceNumber > 0){
		extraTime = atof(segs[1].c_str());
		serverExecutionTime = atof(segs[2].c_str());
		N1 = atof(segs[3].c_str());
		N2 = totalProcessingTime - N1 - serverExecutionTime;
	}
	for (int i = 4; i < (4 + faceNumber); ++i){
		std::vector<std::string> faceSegs = Utility::split(segs[i], ",");
		/**
		if (faceSegs.size() !=  20){
			std::cout << "Problematic string:" << segs[i] << " size:"<< faceSegs.size() << std::endl;
			exit (EXIT_FAILURE);
		}**/
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
		int featureNum = atoi(faceSegs[8].c_str());
		std::vector<cv::Point2f> featurePoints;
		for (int j = 0; j < featureNum; ++j){
			int index = (2*j) + 9;
			featurePoints.push_back(cv::Point2f(atof(faceSegs[index].c_str()), atof(faceSegs[index+1].c_str())));
		}
		

		std::vector<std::string> probStr = Utility::split(segs[i], ";");
		std::vector<std::string> probSegs = Utility::split(probStr[1], ",");

		std::vector<double> probEstimate(225, 0.0);
		for (int j = 0; j < probSegs.size()/2; ++j){
			int index = 2 * j;
			int l = atoi(probSegs[index].c_str());
			double p = atof(probSegs[index + 1].c_str());
			probEstimate.at(l) = p;
		}

		FaceClass face(faceID, label, "later", confidence, left, top, width, height, wantMore, featurePoints, probEstimate);
		faces.push_back(face);	
	}
	if (faces.size() != faceNumber){
		std::cout << "Problematic response string:" << response_str << std::endl;
		exit (EXIT_FAILURE);
	}

}
