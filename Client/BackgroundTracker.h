#ifndef BACKGROUNDTRACKER_H
#define BACKGROUNDTRACKER_H
#include <opencv2/highgui/highgui.hpp>
#include <deque>
#include "FaceClass.h"

class BackgroundTracker{

private:
	
	int maxFeatures;
	double qualityLevel;
	double minDistance;
	int blockSize;
	bool useHarrisDetector;
	double k;
	cv::Size subPixWinSize, winSize;
	cv::TermCriteria termcrit;
	int maxLevel;
	int flags;
	double minEigThreshold;
	

public:
	
	cv::Mat prevFrame;
	std::vector<cv::Point2f> prevFeatures;

	BackgroundTracker()
		: maxFeatures(100)
		, qualityLevel(0.1)
		, minDistance(5)
		, blockSize(3)
		, useHarrisDetector(false)
		, subPixWinSize(10,10)
		, termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 30, 0.01)
		, flags(0)
		, k(0.04)
		{
		}
	~BackgroundTracker(){
		prevFeatures.clear();
		prevFrame.release();
	};

	bool BackgroundTracker::isLost(cv::Rect& lostRegion);
	int track(cv::Mat& curFrame, int& executionTime);
	void BackgroundTracker::init(cv::Mat& curFrame);

};
#endif