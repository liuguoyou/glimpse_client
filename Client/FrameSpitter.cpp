#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "FrameSpitter.h"
#include "Utility.h"
#include <string>
#include <iostream>

FrameSpitter::FrameSpitter(std::string _videoFrameFolder, int _frameRate){
	videoFrameFolder = _videoFrameFolder;
	frameRate = _frameRate;
	loadFrames();
}


void FrameSpitter::loadFrames(){

	frames = Utility::list_files(videoFrameFolder);
	totalFrameNum = (int)frames.size();
	
	
	//for (int i = 0; i < totalFrameNum; ++i){
	//	std::cout << frames.at(i) << std::endl;
	//}
	
	
}

bool FrameSpitter::has_frame(double time){

	int frameIndex = static_cast<int> (time / (1000 / (frameRate * 1.0)));	
	if (frameIndex < totalFrameNum){
		return true;
	}
	return false;
}


cv::Mat FrameSpitter::spit(double time, int &frameIndex, double &captureTime, std::string& frameName){
	cv::Mat frame;
	frameIndex = static_cast<int> (time / (1000 / (frameRate * 1.0)));	
	captureTime = (double)frameIndex * (1000 / (frameRate * 1.0));
	//std::cout << "spitter:" << frameIndex << ";" << totalFrameNum << std::endl;
	while (frameIndex < totalFrameNum){
		frameName = frames.at(frameIndex);		
		std::cout << frameName << std::endl;

		frame = cv::imread(videoFrameFolder + "/" + frameName, CV_8UC1);	
		//frame = cv::imread(videoFrameFolder + "/" + frameName);	
		
		if (!frame.empty()){
			break;
		}else{
			std::cout << "Warning: Jump one frame" << std::endl;
			++frameIndex;
		}
	}
	return frame;
}