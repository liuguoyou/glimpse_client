#include "BackgroundTracker.h"
#include "Utility.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <iostream>
#include <windows.h>
#include <random>

inline float Min(float a, float b) {return a < b ? a : b;}
inline float Max(float a, float b) {return a > b ? a : b;}




void BackgroundTracker::init(cv::Mat& curFrame){

	for(int x_i = 0; x_i < curFrame.cols ; x_i += 40){
		for (int y_i = 0; y_i < curFrame.rows ; y_i += 20){
			cv::Point2f point( x_i + 10, y_i + 10);
			prevFeatures.push_back(point);
		}
	}
	curFrame.copyTo(prevFrame);
}


int estimateExecutionTime(cv::Mat frame){
	
/** for 100 feature points
0.1	16.59009413	4.464659221
0.2	27.00416406	6.12319791
0.3	38.53606142	7.665464279
0.4	40.73091879	7.561363837
0.5	43.43406001	7.659522702
0.6	56.62027997	9.376144228
0.7	59.83267033	9.506935717
0.8	63.83562156	9.76369012
0.9	68.49970398	10.0998235
1	73.56380535	10.50580556

**/
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(73.56380535, 10.50580556);
	
	double number = distribution(generator);
	return (int)number;

}
bool xSort(cv::Point2f a, cv::Point2f b){
	
	return (a.x < b.x);
}

bool ySort(cv::Point2f a, cv::Point2f b){
	
	return (a.y < b.y);
}
bool BackgroundTracker::isLost(cv::Rect& trackedRegion){
	
	/**
	std::sort(prevFeatures.begin(), prevFeatures.end(), xSort);
	float q1_x = prevFeatures.at(prevFeatures.size() /4).x;
	float q2_x = prevFeatures.at(prevFeatures.size() /2).x;
	float q3_x = prevFeatures.at(prevFeatures.size() /2 + prevFeatures.size() /4).x;

	float interval_x = (q3_x - q1_x) * 1.5;

	std::sort(prevFeatures.begin(), prevFeatures.end(), ySort);
	float q1_y = prevFeatures.at(prevFeatures.size() /4).y;
	float q2_y = prevFeatures.at(prevFeatures.size() /2).y;
	float q3_y = prevFeatures.at(prevFeatures.size() /2 + prevFeatures.size() /4).y;
	float interval_y = (q3_y - q1_y) * 1.5;

	cv::Point2f min_point(FLT_MAX, FLT_MAX);
	cv::Point2f max_point(FLT_MIN, FLT_MIN);

	min_point.x = q1_x - interval_x;
	min_point.y = q1_y - interval_y;
	max_point.x = q3_x + interval_x;
	max_point.y = q3_y + interval_y;
	**/
	cv::Point2f min_point(FLT_MAX, FLT_MAX);
	cv::Point2f max_point(FLT_MIN, FLT_MIN);
	for (int i = 0; i < prevFeatures.size(); ++i){
		min_point.x = Min(min_point.x, prevFeatures.at(i).x);
		min_point.y = Min(min_point.y, prevFeatures.at(i).y);
		max_point.x = Max(max_point.x, prevFeatures.at(i).x);
		max_point.y = Max(max_point.x, prevFeatures.at(i).y);
	}
	min_point.x += 20;
	min_point.y += 20;
	max_point.x -= 20;
	max_point.y -= 20;
	if (min_point.x < 0) min_point.x = 0;
	if (min_point.y < 0) min_point.y = 0;
	if (max_point.x > prevFrame.cols) max_point.x = prevFrame.cols;
	if (max_point.y > prevFrame.rows) max_point.y = prevFrame.rows;

	trackedRegion.x = min_point.x;
	trackedRegion.y = min_point.y;
	trackedRegion.width = max_point.x - min_point.x;
	trackedRegion.height = max_point.y - min_point.y;

	if ((trackedRegion.width * trackedRegion.height) < (prevFrame.cols * prevFrame.rows * 1 / 2.0)){
		std::cout << "isLOST!!!!!" << std::endl;
		return true;
	}
	
	return false;
}
int BackgroundTracker::track(cv::Mat& _curFrame, int& executionTime){ 
	
	int returnValue;
	cv::vector<float> err;
	cv::vector<uchar> status;
	std::vector<cv::Point2f> curFeatures;
	cv::calcOpticalFlowPyrLK(prevFrame, _curFrame, prevFeatures, curFeatures,
		   status, err, cv::Size(31,31), 3, termcrit, 0, 0.01);
	
	size_t i, k;
	for(i = k = 0; i < curFeatures.size(); ++i){		

		// status[i] = 0, the feature has lost
		if (status[i] == 0 || Utility::myDistance(curFeatures[i].x, curFeatures[i].y , prevFeatures[i].x , prevFeatures[i].y) > 50){
			continue;
		}
	
		curFeatures[k].x = curFeatures[i].x;
		curFeatures[k].y = curFeatures[i].y;
		
		cv::circle(_curFrame, curFeatures[k], 2, CV_RGB(255,0,0),1);
		
		++k;
	}
	curFeatures.resize(k);
	
	//cv::imshow("backgroundTrack", _curFrame);
	//cv::waitKey(60);
	// estimate execution time //
	executionTime = estimateExecutionTime(prevFrame);
	
	
	// update //
	_curFrame.copyTo(prevFrame);
	prevFeatures = curFeatures;
	
	return 0;
}