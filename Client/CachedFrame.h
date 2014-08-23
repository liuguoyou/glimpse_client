#ifndef CACHEFRAMECLASS_H
#define CACHEFRAMECLASS_H
#include <opencv2/highgui/highgui.hpp>

class CachedFrame{
public:
	cv::Mat frame;
	double timeStamp;
	std::string frameName;


	CachedFrame(cv::Mat& _frame, double _timeStamp, std::string _frameName){
		frame = _frame;
		timeStamp = _timeStamp;
		frameName = _frameName;
	};
};
#endif