#include "MotionAnalyzer.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

inline float Min(float a, float b) {return a < b ? a : b;}
inline float Max(float a, float b) {return a > b ? a : b;}

bool MotionAnalyzer::checkMoving(cv::Mat curFrame){
	cv::Rect movingRegion;
	if (prevFrame.cols == 0 || prevFrame.rows == 0){
			
			movingRegion.x = 0;
			movingRegion.y = 0;
			movingRegion.width = curFrame.cols;
			movingRegion.height = curFrame.rows;

			return true;
	}else{

			cv::Mat diff;
			cv::absdiff(curFrame, prevFrame, diff);
			cv::Mat bin; // get the pixels that correspond to areas of motion
			
			cv::Point2f min(FLT_MAX, FLT_MAX);
			cv::Point2f max(FLT_MIN, FLT_MIN);
			
			
			int numOfPixelsMoving = 0;
			//cv::threshold(diff, bin, THRESHOLD, 255, 0);
			for (int i = 0; i < diff.rows; ++i){
				for (int j = 0; j < diff.cols; ++j){
					int index = i * diff.cols + j;
					if (diff.data[index] > THRESHOLD){
						//++numOfPixelsMoving;
						min.x = Min(j, min.x);
						min.y = Min(i, min.y);
						max.x = Max(j, max.x);
						max.y = Max(i, max.y);
					}
				}
			}
			movingRegion.x = cvRound(min.x);
			movingRegion.y = cvRound(min.y);
			movingRegion.width = cvRound((max.x - min.x));
			movingRegion.height = cvRound((max.y - min.y));
	}
		if (movingRegion.width < 20 || movingRegion.height < 20 ){
				return false;
		}else{
				return true;
		}

}

bool MotionAnalyzer::isMoving(cv::Mat& curFrame, cv::Rect& movingRegion){

		if (prevFrame.cols == 0 || prevFrame.rows == 0){
			// we don't have a prevframe
			curFrame.copyTo(prevFrame);
			movingRegion.x = 0;
			movingRegion.y = 0;
			movingRegion.width = curFrame.cols;
			movingRegion.height = curFrame.rows;

			return true;

		}else{

			cv::Mat diff;
			cv::absdiff(curFrame, prevFrame, diff);
			cv::Mat bin; // get the pixels that correspond to areas of motion
			
			cv::Point2f min(FLT_MAX, FLT_MAX);
			cv::Point2f max(FLT_MIN, FLT_MIN);
			
			
			int numOfPixelsMoving = 0;
			cv::threshold(diff, bin, THRESHOLD, 255, 0);
			for (int i = 0; i < diff.rows; ++i){
				for (int j = 0; j < diff.cols; ++j){
					int index = i * diff.cols + j;
					if (diff.data[index] > THRESHOLD){
						++numOfPixelsMoving;
						min.x = Min(j, min.x);
						min.y = Min(i, min.y);
						max.x = Max(j, max.x);
						max.y = Max(i, max.y);
					}
				}
			}
			std::cout << "final:" << min.x << "," << min.y << "," << max.x-min.x << "," << max.y-min.y<< "," << numOfPixelsMoving << std::endl; 
			movingRegion.x = cvRound(min.x);
			movingRegion.y = cvRound(min.y);
			movingRegion.width = cvRound((max.x - min.x));
			movingRegion.height = cvRound((max.y - min.y));

#ifdef _DEBUG
			/**
			std::string str = "false";
			if (numOfPixelsMoving >  1.3 * diff.cols * diff.rows){
				str = "true";
			}
			
			cv::putText(bin,str,cv::Point(20,20),1, 2, CV_RGB(0,0,255), 1, 8, false);
			rectangle(bin, cvPoint(movingRegion.x, movingRegion.y), cvPoint((movingRegion.x + movingRegion.width), 
				movingRegion.y + movingRegion.height), CV_RGB(0,0,255), 3, 8, 0);
			
			cv::imshow( "diff image", bin );
			cv::waitKey( 100 );
			getchar();
			**/
#endif
			curFrame.copyTo(prevFrame);
			
			if (numOfPixelsMoving > 5000){
				return true;
			}else{
				return false;
			}
		}
	}