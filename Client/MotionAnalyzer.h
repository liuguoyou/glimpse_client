#ifndef MOTIONANALYZER_H
#define MOTIONANALYZER_H

#include <opencv2/highgui/highgui.hpp>
class MotionAnalyzer{

private:

	cv::Mat prevFrame;

public:
	int THRESHOLD;

	MotionAnalyzer()
	:prevFrame(0, 0, CV_8UC1),
	THRESHOLD(80)
	{
	}
	bool checkMoving(cv::Mat curFrame);
	bool isMoving(cv::Mat& curFrame, cv::Rect& movingRegion);


};

#endif