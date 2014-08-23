#include "FaceTracker.h"
#include "Utility.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <iostream>
#include <windows.h>
#include <random>

inline float Min(float a, float b) {return a < b ? a : b;}
inline float Max(float a, float b) {return a > b ? a : b;}


// Initialize tracker
size_t FaceTracker::init(cv::Mat& curFrame,  cv::Rect& faceRect){

	cv::Mat mask = cv::Mat::zeros(curFrame.rows, curFrame.cols, CV_8U);
	mask(faceRect) = 1;
	std::vector<cv::Point2f> featurePoints;
	cv::goodFeaturesToTrack(curFrame, 
							featurePoints,
							maxFeatures,
							qualityLevel,
							minDistance,
							mask,
							blockSize,
							useHarrisDetector,
							k);
	cv::cornerSubPix(curFrame, featurePoints, subPixWinSize, cv::Size(-1,-1), termcrit);	
	
	curFrame.copyTo(prevFrame);
	_prevFeatures = featurePoints;
	return featurePoints.size();
}

void FaceTracker::init(FaceClass& _fc, cv::Mat& curFrame, double _startTime){
	// using the salient points from the server
	startTime = _startTime;
	fc = _fc;
	_prevFeatures.push_back(cv::Point2f(fc.faceRect.x, fc.faceRect.y));
	_prevFeatures.push_back(cv::Point2f(fc.faceRect.x + fc.faceRect.width, fc.faceRect.y));
	_prevFeatures.push_back(cv::Point2f(fc.faceRect.x, fc.faceRect.y + fc.faceRect.height));
	_prevFeatures.push_back(cv::Point2f(fc.faceRect.x + fc.faceRect.width, fc.faceRect.y + fc.faceRect.height));
	//_prevFeatures.push_back(cv::Point2f((fc.faceRect.x + fc.faceRect.width)/2, (fc.faceRect.y + fc.faceRect.height)/2));

	/**
	for (int i = 0; i < fc.featurePoints.size(); ++i){
		_prevFeatures.push_back(cv::Point2f(fc.featurePoints.at(i).x, fc.featurePoints.at(i).y));
	}
	**/
	
	faceTrajectroy.push_back(fc.faceRect);
	curFrame.copyTo(prevFrame);
}
void FaceTracker::init(FaceClass& _fc, cv::Mat& curFrame){
	// using the salient points from the server
	fc = _fc;
	_prevFeatures.push_back(cv::Point2f(fc.faceRect.x, fc.faceRect.y));
	_prevFeatures.push_back(cv::Point2f(fc.faceRect.x + fc.faceRect.width, fc.faceRect.y));
	_prevFeatures.push_back(cv::Point2f(fc.faceRect.x, fc.faceRect.y + fc.faceRect.height));
	_prevFeatures.push_back(cv::Point2f(fc.faceRect.x + fc.faceRect.width, fc.faceRect.y + fc.faceRect.height));
	/**
	_prevFeatures.push_back(cv::Point2f(fc.faceRect.x + fc.faceRect.width/2, fc.faceRect.y + fc.faceRect.height/2));
	
	_prevFeatures.push_back(cv::Point2f(fc.featurePoints.at(0).x, fc.featurePoints.at(0).y));
	
	_prevFeatures.push_back(cv::Point2f(fc.featurePoints.at(1).x, fc.featurePoints.at(1).y));
	
	_prevFeatures.push_back(cv::Point2f(fc.featurePoints.at(2).x, fc.featurePoints.at(2).y));
	
	_prevFeatures.push_back(cv::Point2f(fc.featurePoints.at(3).x, fc.featurePoints.at(3).y));
	
	_prevFeatures.push_back(cv::Point2f(fc.featurePoints.at(4).x, fc.featurePoints.at(4).y));
	**/
	/**
	for (int i = 0; i < fc.featurePoints.size(); ++i){
		_prevFeatures.push_back(cv::Point2f(fc.featurePoints.at(i).x, fc.featurePoints.at(i).y));
	}**/
	
	
	faceTrajectroy.push_back(fc.faceRect);
	curFrame.copyTo(prevFrame);
}

int distance(int x1, int y1, int x2, int y2){
	return sqrt((x1-x2) * (x1-x2) + (y1-y2) * (y1-y2));
}

double estimateExecutionTime(cv::Rect ROI, cv::Mat frame){
	
/** for 30 feature points
0.1	5.974168837	1.73664996
0.2	9.379017105	2.349806995
0.3	14.3536107	2.989265104
0.4	16.60818501	3.184199667
0.5	18.97701342	3.63101768
0.6	25.70510134	4.738983804
0.7	29.51849074	5.389864823
0.8	33.62009476	6.216371195
0.9	37.92186916	7.161423506
1	42.29251578	7.351960796
**/
	// 30 points 
	//float mean[] = {0, 5.974168837,9.379017105,14.3536107,16.60818501,18.97701342,25.70510134,29.51849074,33.62009476,37.92186916,42.29251578};	//float std [] = {0,1.73664996,2.349806995,2.989265104,3.184199667,3.63101768,4.738983804,5.389864823,6.216371195,7.161423506,7.351960796};
	
	//10 points
	//float mean[] = {0, 3.233856434,5.601501451,8.642235604,11.19284076,13.53128656,18.85999821,22.69797813,27.3288545,32.30271112,37.68275257};	//float std [] = {0, 1.51915146,1.834284376,2.869639845,3.571048152,4.806804306,6.349396926,7.573714244,8.636492883,9.943047802,11.50088856};
	// 5 points
	float mean[] = {0, 1.551237568,3.519289161, 7.464443928, 10.20457988, 13.45311379, 16.40693511,20.22669785,25.00923833,30.0843427,35.36997682};	
	float std[] = {0, 0.826090404, 1.40336688578, 2.05513542316, 2.77236678357, 3.23751949696, 4.03336732085, 4.55603943125, 5.07848422031, 6.05924497718,  7.24563557259};
	float ROIResolution = ROI.width * ROI.height;
	int ratio = 0;
	int ratioBin = cvRound(10 * sqrt(ROIResolution / float(frame.cols * frame.rows)));
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(mean[ratioBin],std[ratioBin]);
	
	double number = distribution(generator);
	std::cout << "normal dist: " << ratioBin << "," << mean[ratioBin] << "," << std[ratioBin] << number << std::endl; 
	return number;
}
bool myFunction(cv::Point2f a, cv::Point2f b){
	float a_v = sqrt((a.x * a.x) + (a.y * a.y));
	float b_v = sqrt((b.x * b.x) + (b.y * b.y));
	return (a_v < b_v);
}

int FaceTracker::trackWholeFrame(cv::Mat& _curFrame, std::vector<cv::Point2f>& _curFeatures, cv::Rect& faceRect, double& executionTime){


	int returnValue;
	cv::vector<float> err;
	cv::vector<uchar> status;
	cv::calcOpticalFlowPyrLK(prevFrame, _curFrame, _prevFeatures, _curFeatures,
		   status, err, cv::Size(31,31), 3, termcrit, 0, 0.001);

	cv::Point2f min_point(FLT_MAX, FLT_MAX);
	cv::Point2f max_point(FLT_MIN, FLT_MIN);
	// refactor the points array to remove points lost due to tracking error, 
	// and map it to the original image location
	size_t i, k;
	for (i = k = 0; i < _curFeatures.size(); ++i){
		
		// status[i] = 0, the feature has lost
		//if (status[i] == 0 || distance(_curFeatures[i].x, _curFeatures[i].y, _prevFeatures[i].x, _prevFeatures[i].y) > 30){
		if (status[i] == 0){
			continue;
		}

		// adjust it to the correct position
		_curFeatures[k].x = _curFeatures[i].x;
		_curFeatures[k].y = _curFeatures[i].y;

		min_point.x = Min(min_point.x, _curFeatures[k].x);
		min_point.y = Min(min_point.y, _curFeatures[k].y);
		max_point.x = Max(max_point.x, _curFeatures[k].x);
		max_point.y = Max(max_point.y, _curFeatures[k].y);
		
		++k;
	}
	_curFeatures.resize(k);
	
	// stablize and decide the bounding box
	faceRect.x = cvRound(min_point.x);
	faceRect.y = cvRound(min_point.y);
	faceRect.width = cvRound((max_point.x - min_point.x));
	faceRect.height = cvRound((max_point.y - min_point.y));
	fc.faceRect = faceRect;
	fc.featurePoints = _curFeatures;

	//if (faceRect.width < 25 || faceRect.height < 25 || k < 15 || faceRect.height/(faceRect.width * 1.0) > 2.5){
	if (faceRect.width < 25 || faceRect.height < 25 || k < 15){	
		return -1;
	}


	// estimate time
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(42.29251578,7.351960796);
	executionTime = distribution(generator);
	
	// update //
	_curFrame.copyTo(prevFrame);
	_prevFeatures = _curFeatures;
	

	return 0;

}

int FaceTracker::subRegionTrack(cv::Mat& _curFrame, std::vector<cv::Point2f>& _curFeatures, cv::Rect& faceRect, double& executionTime){ 
	
	
	// do tracking in the subregion, we always have at least one history //
	cv::Rect prevRect = faceTrajectroy.at(faceTrajectroy.size()-1);
	int roix = (prevRect.x - prevRect.width) > 0 ? (prevRect.x - prevRect.width):0;
	int roiy = (prevRect.y - prevRect.height) > 0 ? (prevRect.y - prevRect.height):0;
	int roiW = (roix + (3 * prevRect.width)) <= _curFrame.cols? (3 * prevRect.width) :  (_curFrame.cols - roix);
	int roiH = (roiy + (3 * prevRect.height)) <= _curFrame.rows? (3 * prevRect.height) :  (_curFrame.rows - roiy);
	cv::Rect ROI( roix, roiy, roiW, roiH);
	
	// crop original image, update feature points // 
	cv::Mat croppedPrevFrame(prevFrame, ROI);
	cv::Mat croppedCurFrame(_curFrame, ROI);
	std::vector<cv::Point2f> croppedPrevFeatures;
	for (int i = 0 ; i < _prevFeatures.size(); ++i){
		cv::Point2f p(_prevFeatures.at(i).x - ROI.x, _prevFeatures.at(i).y - ROI.y);
		croppedPrevFeatures.push_back(p);
	}

	
	
	// Track on subregion 
	int returnValue;
	cv::vector<float> err;
	cv::vector<uchar> status;
	std::vector<cv::Point2f> croppedCurFeatures;
	cv::calcOpticalFlowPyrLK(croppedPrevFrame, croppedCurFrame, croppedPrevFeatures, croppedCurFeatures,
		   status, err, cv::Size(31,31), 3, termcrit, 0, 0.001);
	
	
	
	cv::Point2f min_point(FLT_MAX, FLT_MAX);
	cv::Point2f max_point(FLT_MIN, FLT_MIN);
	// refactor the points array to remove points lost due to tracking error, 
	// and map it to the original image location
	size_t i, k;
	for (i = k = 0; i < croppedCurFeatures.size(); ++i){
		
		// status[i] = 0, the feature has lost
		//if (status[i] == 0){
		//	continue;
		//}
		if (status[i] == 0 || distance(croppedCurFeatures[i].x, croppedCurFeatures[i].y , croppedPrevFeatures[i].x , croppedPrevFeatures[i].y) > 30){
			continue;
		}

		croppedCurFeatures[k].x = croppedCurFeatures[i].x + ROI.x;
		croppedCurFeatures[k].y = croppedCurFeatures[i].y + ROI.y;

		min_point.x = Min(min_point.x, croppedCurFeatures[k].x);
		min_point.y = Min(min_point.y, croppedCurFeatures[k].y);
		max_point.x = Max(max_point.x, croppedCurFeatures[k].x);
		max_point.y = Max(max_point.y, croppedCurFeatures[k].y);
		
		++k;
	
	}
	croppedCurFeatures.resize(k);
	_curFeatures = croppedCurFeatures;



	// stablize and decide the bounding box
	faceRect.x = cvRound(min_point.x);
	faceRect.y = cvRound(min_point.y);
	faceRect.width = cvRound((max_point.x - min_point.x));
	faceRect.height = cvRound((max_point.y - min_point.y));

	fc.faceRect = faceRect;
	fc.featurePoints = _curFeatures;

	if (faceRect.width < 25 || faceRect.height < 25 || k < 15 || faceRect.height/(faceRect.width * 1.0) > 2.5){
		return -1;
	}
	// estimate execution time //
	executionTime = estimateExecutionTime(ROI, prevFrame);
	
	// save into history //
	faceTrajectroy.push_back(faceRect);
	if (faceTrajectroy.size() == HISTORY_NUM + 1){
		faceTrajectroy.pop_front();
	}

	// update //
	_curFrame.copyTo(prevFrame);
	_prevFeatures = _curFeatures;
	
	return 0;
}

int FaceTracker::track(cv::Mat& _curFrame, std::vector<cv::Point2f>& _curFeatures, cv::Rect& faceRect, double& executionTime){ 
	
	
	// do tracking in the subregion, we always have at least one history //
	cv::Rect prevRect = faceTrajectroy.at(faceTrajectroy.size()-1);
	int roix = (prevRect.x - 1.5 * prevRect.width) > 0 ? (prevRect.x - 1.5 * prevRect.width):0;
	int roiy = (prevRect.y - 1.5 * prevRect.height) > 0 ? (prevRect.y - 1.5 * prevRect.height):0;
	int roiW = (roix + (4 * prevRect.width)) <= _curFrame.cols? (4 * prevRect.width) :  (_curFrame.cols - roix);
	int roiH = (roiy + (4 * prevRect.height)) <= _curFrame.rows? (4 * prevRect.height) :  (_curFrame.rows - roiy);
	cv::Rect ROI( roix, roiy, roiW, roiH);
	//std::cout << "ROI" <<  ROI.x << "," << ROI.y << "," << ROI.width << "," << ROI.height <<std::endl;
	/**
	if (faceTrajectroy.size() > 1){
		cv::Rect faceVelocity;
		float alpha = 0.7;
		for (int i = 1 ; i < faceTrajectroy.size(); ++i){
			// estimate velocity
			cv::Rect pRect = faceTrajectroy.at(i-1);
			cv::Rect cRect = faceTrajectroy.at(i);

			if (i == 1){
				faceVelocity.x = cRect.x - pRect.x;
				faceVelocity.y = cRect.y - pRect.y;
				faceVelocity.width = cRect.width - pRect.width;
				faceVelocity.height = cRect.height - pRect.height;
			}else{
				faceVelocity.x = alpha * (cRect.x - pRect.x) + (1-alpha) * faceVelocity.x;
				faceVelocity.y = alpha * (cRect.y - pRect.y) + (1-alpha) * faceVelocity.y;
				faceVelocity.width = alpha * (cRect.width - pRect.width) + (1-alpha) * faceVelocity.width;
				faceVelocity.height = alpha * (cRect.height - pRect.height) + (1-alpha) * faceVelocity.height;
			}
			
		}
		//update ROI
		ROI.x = prevRect.x + faceVelocity.x;
		ROI.y = prevRect.y + faceVelocity.y;
		ROI.width = prevRect.width + faceVelocity.width;
		ROI.height = prevRect.height + faceVelocity.height;
	}
	**/
	// crop original image, update feature points // 
	cv::Mat croppedPrevFrame(prevFrame, ROI);
	cv::Mat croppedCurFrame(_curFrame, ROI);
	std::vector<cv::Point2f> croppedPrevFeatures;
	for (int i = 0 ; i < _prevFeatures.size(); ++i){
		cv::Point2f p(_prevFeatures.at(i).x - ROI.x, _prevFeatures.at(i).y - ROI.y);
		croppedPrevFeatures.push_back(p);
	}

	/**
	for (int i = 0; i < croppedPrevFeatures.size(); ++i){
		circle( croppedPrevFrame, croppedPrevFeatures[i], 3, cv::Scalar(0,255,0), -1 , 8);
	}
	cv::imshow("croppedFrame", croppedPrevFrame);
	cv::waitKey( 50 );
	**/
	// Track on subregion 
	int returnValue;
	cv::vector<float> err;
	cv::vector<uchar> status;
	std::vector<cv::Point2f> croppedCurFeatures;
	cv::calcOpticalFlowPyrLK(croppedPrevFrame, croppedCurFrame, croppedPrevFeatures, croppedCurFeatures,
		   status, err, cv::Size(31,31), 3, termcrit, 0, 0.001);
	
	//cv::calcOpticalFlowPyrLK(prevFrame, _curFrame, _prevFeatures, _curFeatures,
	//	   status, err, cv::Size(31,31), 3, termcrit, 0, 0.001);
	
	cv::Point2f min_point(FLT_MAX, FLT_MAX);
	cv::Point2f max_point(FLT_MIN, FLT_MIN);
	// refactor the points array to remove points lost due to tracking error, 
	// and map it to the original image location
	size_t i, k;
	std::vector<cv::Point2f> velocity;
	for (i = 0; i < croppedCurFeatures.size(); ++i){
		// adjust it to the correct position
		croppedCurFeatures.at(i).x +=  ROI.x;
		croppedCurFeatures.at(i).y +=  ROI.y;
		
	
		// status[i] = 0, the feature has lost
		//if (status[i] == 0 || distance(croppedCurFeatures[i].x, croppedCurFeatures[i].y , _prevFeatures[i].x , _prevFeatures[i].y) > 30){
		if (status[i] == 0){
			
		continue;
		}
		
		// compute the moving speed for each of the good feature point //
		cv::Point2f vp(croppedCurFeatures.at(i).x - _prevFeatures.at(i).x, croppedCurFeatures.at(i).y - _prevFeatures.at(i).y);
		velocity.push_back(vp); 
	}
	// tracker failed!
	/**
	if (velocity.size() < 3)
		return -1;
	
	std::sort(velocity.begin(), velocity.end(), myFunction);
	int mid = velocity.size()/2;
	int offset = velocity.size() % 2 == 0? mid: (mid+1); 
	float q1, q3 = 0;
	if (mid % 2 == 0){
		int q1_index = (mid/2 + 1) >= velocity.size()? (velocity.size()-1) :(mid/2 + 1);
		q1 = (sqrt(velocity.at((mid/2)).x * velocity.at((mid/2)).x + velocity.at((mid/2)).y * velocity.at((mid/2)).y) + 
		sqrt(velocity.at(q1_index).x * velocity.at(q1_index).x + velocity.at(q1_index).y * velocity.at(q1_index).y))/2;

		int q3_index_1 = (mid/2) + offset >= velocity.size()? velocity.size()-1:(mid/2) + offset;
		int q3_index_2 = (mid/2)+ 1 + offset >= velocity.size()? velocity.size()-1:(mid/2) + offset +1 ;

		q3 = (sqrt(velocity.at(q3_index_1).x * velocity.at(q3_index_1).x + velocity.at(q3_index_1).y * velocity.at(q3_index_1).y) + 
		sqrt(velocity.at(q3_index_2).x * velocity.at(q3_index_2).x + velocity.at(q3_index_2).y * velocity.at(q3_index_2).y))/2;
	
	}else{
		int q3_index = (mid/2) + offset >= velocity.size()? velocity.size()-1:(mid/2) + offset;
		q1 = sqrt(velocity.at((mid/2)).x * velocity.at((mid/2)).x + velocity.at((mid/2)).y * velocity.at((mid/2)).y);
		q3 = sqrt(velocity.at(q3_index).x * velocity.at(q3_index).x + velocity.at(q3_index).y * velocity.at(q3_index).y);
	}
	
	float interval = (q3 - q1) * 3;
	**/

	for(i = k = 0; i < croppedCurFeatures.size(); ++i){		

		// status[i] = 0, the feature has lost
		if (status[i] == 0){
			continue;
		}
		// for each working feature point //
		/**
		cv::Point2f vp(croppedCurFeatures.at(i).x - _prevFeatures.at(i).x, croppedCurFeatures.at(i).y - _prevFeatures.at(i).y);
		float v = sqrt(vp.x * vp.x) + (vp.y * vp.y);
		if (v  > q3 + interval){
			croppedCurFeatures.at(i).x = _prevFeatures.at(i).x + velocity.at(mid).x;
			croppedCurFeatures.at(i).y = _prevFeatures.at(i).y + velocity.at(mid).y;
		}
		**/
		croppedCurFeatures[k].x = croppedCurFeatures[i].x;
		croppedCurFeatures[k].y = croppedCurFeatures[i].y;

		min_point.x = Min(min_point.x, croppedCurFeatures[k].x);
		min_point.y = Min(min_point.y, croppedCurFeatures[k].y);
		max_point.x = Max(max_point.x, croppedCurFeatures[k].x);
		max_point.y = Max(max_point.y, croppedCurFeatures[k].y);
		
		++k;
	}
	croppedCurFeatures.resize(k);
	_curFeatures = croppedCurFeatures;



	// stablize and decide the bounding box
	faceRect.x = cvRound(min_point.x);
	faceRect.y = cvRound(min_point.y);
	//faceRect.width = (cvRound((max_point.x - min_point.x)) +  cvRound((max_point.y - min_point.y)))/2;
	//faceRect.height = (cvRound((max_point.x - min_point.x)) +  cvRound((max_point.y - min_point.y)))/2;
	faceRect.width = cvRound(max_point.x - min_point.x);
	faceRect.height = cvRound(max_point.y - min_point.y);

	fc.faceRect = faceRect;
	fc.featurePoints = _curFeatures;

//	if (faceRect.width < 25 || faceRect.height < 25 || k < 15){
	if (faceRect.width < 25 || faceRect.height < 25 || k < 3){
	
		std::cout << "tracker fails because of width/height/k:" << faceRect.width << "," << faceRect.height << "," << k <<  std::endl;
		return -1;
	}

	// estimate execution time //
	executionTime = estimateExecutionTime(ROI, prevFrame);
	
	// save into history //
	faceTrajectroy.push_back(faceRect);
	if (faceTrajectroy.size() == HISTORY_NUM + 1){
		faceTrajectroy.pop_front();
	}

	// update //
	_curFrame.copyTo(prevFrame);
	_prevFeatures = _curFeatures;
	
	return 0;
}