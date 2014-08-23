#ifndef UTILITY_H
#define UTILITY_H
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>

class Utility{
public:
	static std::vector<std::string> Utility::list_files(std::string dirName);	
	static double getCurrentTimeMS();
	static double getCurrentTimeMSV2();
	static std::vector<std::string> Utility::split(std::string str, std::string delim);
	static bool wayToSort(std::string i, std::string j);
	static void convertJpegEncodedBuffToMat(char* buffer, int w, int h, cv::Mat& decodedImg);
	static std::map<std::string,int> ReadSubjs(std::string filePath, std::vector<std::string>& labelToSubj);
	static void enlargeROI(cv::Mat curFrame, cv::Rect suggestROI, cv::Mat& croppedEnlargedROI, cv::Rect& enlargedROI);
	static double Utility::myDistance(double x1, double y1, double x2, double y2);
	static int Utility::getCompressionTime(int width, int height);
	static int Utility::getMotionAnalysisTime();
	static double Utility::commonApBPortion(cv::Rect a, cv::Rect b);
	static double Utility::commonAuBPortion(cv::Rect a, cv::Rect b);
	static int Utility::overLap(cv::Rect a, cv::Rect b);
};
#endif