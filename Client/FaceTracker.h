#ifndef FACETRACKER_H
#define FACETRACKER_H
#include <opencv2/highgui/highgui.hpp>
#include <deque>
#include "FaceClass.h"

class FaceTracker{

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
	FaceClass fc;
	cv::Mat prevFrame;
	std::vector<cv::Point2f> _prevFeatures;
	int FEATURE_NUMBER;
	std::deque<cv::Rect> faceTrajectroy;
	double startTime;
	int HISTORY_NUM;

	FaceTracker()
		: maxFeatures(100)
		, qualityLevel(0.1)
		, minDistance(5)
		, blockSize(3)
		, useHarrisDetector(false)
		, subPixWinSize(10,10)
		, termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 30, 0.01)
		, flags(0)
		, k(0.04)
		, FEATURE_NUMBER(30)
		, HISTORY_NUM(3)
		, fc()
		{
		}

	~FaceTracker(){
		_prevFeatures.clear();
		prevFrame.release();
	};
	size_t init(cv::Mat& curFrame, cv::Rect& faceRect);
	void init(FaceClass& _fc, cv::Mat& curFrame);
	void init(FaceClass& _fc, cv::Mat& curFrame, double _startTime);
	
	int track(cv::Mat& curFrame, std::vector<cv::Point2f>& curFeatures, cv::Rect& faceRect, double& executionTime);
	int FaceTracker::trackWholeFrame(cv::Mat& _curFrame, std::vector<cv::Point2f>& _curFeatures, cv::Rect& faceRect, double& executionTime);
	int FaceTracker::subRegionTrack(cv::Mat& _curFrame, std::vector<cv::Point2f>& _curFeatures, cv::Rect& faceRect, double& executionTime);
	void FaceTracker::init(cv::Mat& curFrame);

};
#endif