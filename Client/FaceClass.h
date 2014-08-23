#ifndef FACECLASS_H
#define FACECLASS_H
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

class FaceClass{
public:
	int faceID;
	int label;
	std::string labelName;
	double confidence;
	cv::Rect faceRect;
	bool wantMore;
	std::vector<cv::Point2f> featurePoints;
	double overlapRatio;
	std::vector<double> ewma;

	FaceClass()
		: faceID(-1)
		, label(-1)
		, labelName("")
		, confidence(-1.0)
		, faceRect(0,0,0,0)
		, wantMore(false)
		, featurePoints()
	{}

	//FaceClass(std::string response_str);
	void updateEWMA(std::vector<double> prediction);
	FaceClass(int _faceID, int _label, std::string _labelName, double _confidence, int left, int top, int width, int height, bool _wantMore, std::vector<cv::Point2f> _featurePoints, std::vector<double> _probEstimate){
		faceID = _faceID;
		label = _label;
		labelName = _labelName;
		confidence = _confidence;
		faceRect = cv::Rect(left, top, width, height);
		wantMore = _wantMore;
		featurePoints = _featurePoints;
		ewma = _probEstimate;
	};

	FaceClass( int _label, int left, int top, int width, int height){
		label = _label;
		faceRect = cv::Rect(left, top, width, height);
		ewma.reserve(225);
		for (int i = 0; i < ewma.size(); ++i){
			ewma.at(i) = 1/225.0;
		}
	};
	
	
	std::string toString(){
		std::stringstream ss;

		ss << faceID << "," << label  << "," << overlapRatio << "," << faceRect.x << "," 
			<< faceRect.y << "," << faceRect.width << "," << faceRect.height << "," << confidence  << "," << wantMore;

		return ss.str();
	};
};

#endif