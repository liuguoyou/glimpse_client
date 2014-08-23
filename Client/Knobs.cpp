#include "Knobs.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void Knobs::cropFrame(const cv::Mat curFrame, cv::Mat& croppedFrame, int x, int y, int width, int height){

	cv::Rect ROI(x, y, width, height);
	cropFrame(curFrame, ROI, croppedFrame);
}

void Knobs::cropFrame(const cv::Mat curFrame, cv::Rect ROI, cv::Mat& croppedFrame){

	cv::Mat(curFrame, ROI).copyTo(croppedFrame);
}
void Knobs::compressFrame(const cv::Mat curFrame, int level, cv::vector<unsigned char>& buffer){
	std::vector<int> params;
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(level);
	
	cv::imencode(".jpg", curFrame, buffer, params);
}
void Knobs::compressFrame(const cv::Mat curFrame, int level, cv::Mat& compressedFrame){
	std::vector<int> params;
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(level);
	std::vector<unsigned char>buffer;
	
	cv::imencode(".jpg", curFrame, buffer, params);
	

	//cv::imwrite("tmp.jpg", curFrame, params);
	//compressedFrame = cv::imread("tmp.jpg", CV_8UC1);

}

void Knobs::resizeFrame(const cv::Mat curFrame, double ratio, cv::Mat& resizedFrame){

	cv::resize(curFrame, resizedFrame, cv::Size(curFrame.cols * ratio, curFrame.rows * ratio));

}