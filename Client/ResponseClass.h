#ifndef RESPONSECLASS_H
#define RESPONSECLASS_H
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "FaceClass.h"

class EntireFrameResponse{
public:
	int processedTime;
	double extraTime;
	double serverExecutionTime;
	double N1;
	double N2;
	int shift_x;
	int shift_y;
	int faceNumber;
	cv::Mat theFrame;
	int transmitTime;
	std::vector<FaceClass> faces;

	EntireFrameResponse()
	: faceNumber(-1)
	, extraTime(-1)
	, serverExecutionTime(-1)
	, theFrame()
	, transmitTime(-1)
	, N2(-1)
	, N1(-1)
	, faces(NULL)
	{}

	EntireFrameResponse(std::string response_str, double totalProcessingTime);

	std::string toString(){
		std::string outStr = std::to_string(faceNumber);
		outStr.append(";");
		for (int i = 0; i < faceNumber; ++i){
			outStr.append(faces[i].toString());
			outStr.append(";");
		}
		outStr.append("\n");
		return outStr;
	};

};
#endif
