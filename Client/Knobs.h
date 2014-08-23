#ifndef KNOBS_H
#define KNOBS_H
#include <opencv2/highgui/highgui.hpp>

class Knobs{
public:
	static void cropFrame(const cv::Mat curFrame, cv::Rect ROI, cv::Mat& croppedFrame);
	static void cropFrame(const cv::Mat curFrame, cv::Mat& croppedFrame, int x, int y, int width, int height);
	static void compressFrame(const cv::Mat curFrame, int level, cv::Mat& compressedFrame);
	static void compressFrame(const cv::Mat curFrame, int level, cv::vector<unsigned char>& compressedFrame);
	static void resizeFrame(const cv::Mat curFrame, double ratio, cv::Mat& resizedFrame);
};

#endif