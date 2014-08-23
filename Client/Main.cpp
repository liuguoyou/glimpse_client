#include <winsock2.h>
#include <windows.h>
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "MotionAnalyzer.h"
#include "FrameSpitter.h"
#include "NetworkService.h"
#include "Utility.h"
#include "ResponseClass.h"
#include "FaceTracker.h"
#include "FrameFaceClass.h"
#include "NativeDetectorWrapper.h"
#include "Knobs.h"
#include "CachedFrame.h"
#include "BackgroundTracker.h"
#include <random>

#define _TRACKING_
using namespace cv;
using namespace std;

enum OperationMode {CLIENTONLY = 0, SERVERONLY = 1, MIX = 2, GRAYMOTIONSERVER =3, SIMPLETRACKING = 4, FORWARDTRACKING = 5, CONTINUETRACKING = 6, IMAGEREGISTRATION = 7, JPEGCOMPRESS = 8, FORWARDTRACKINGEWMA = 9, SLIDE = 10, HARIBASELINE = 11};
const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;

map<string, int> subjs_dict;
int portNum = 8888;
//char* ipAddress = "127.0.0.1";
//char* ipAddress = "128.30.77.95";
char* ipAddress = "192.168.0.1";
//char* ipAddress = "18.95.7.230";
std::vector<FaceTracker> trackers;
int mode = JPEGCOMPRESS; //  use OperationMode
int frameRate = 30;
int trackingState = 0; // 0: get detection frame from server, 1: tracking
double prevSentTime = 0;
int counter = 0;
int COMPRESSLEVEL =	30;
bool isYoutubeDataset = false;
int checkPeriodMs = 1000;
map<string, FrameFaceClass> gtMap;
Mat prevSentFrame;
vector<string> labelToSubj;
string accuracyKey;
bool gotoServer = true;
string sentData = "";
// statistics
int precisionTotal = 0;
int recallTotal = 0;
int comeBackTime = 0;
int correct = 0;
int totalProcessedFrame = 0;
int totalTransmittedByte = 0;
double jpegCompressionTime = 0;
double nextSentTime = 0;
vector<double> aveDelay;
vector<double> aveBoundingBoxDist;
vector<double> hariIntersectArea;
vector<double> boxIntersectArea;
vector<EntireFrameResponse> queuedResponse;
vector<CachedFrame> cachedFrames;
//image registration + motion analysis //
Mat prevProcessedFrame;
BackgroundTracker processedRegionTracker;
Rect trackedRegion;
double nextWholeFrameProcessingTime = 0;
//
int reasonToResend = 0; // 0: counter expire, -1: tracking failed
Rect possibleRect;
int sleepCycle;
string outputFolder;

inline float Min(float a, float b) {return a < b ? a : b;}
inline float Max(float a, float b) {return a > b ? a : b;}


int distance_main(int x1, int y1, int x2, int y2){
	return (x1-x2) * (x1-x2) + (y1-y2) * (y1-y2);
}

void loadGroundTruth(string groundTruthFileName, bool youtubeDataset){
	
	ifstream in;
	in.open(groundTruthFileName);
	string str;
	while ( in ){

		getline(in, str);
		//cout <<str <<endl;
		if (str.length() < 1)
			continue;
		if (youtubeDataset){
			vector<string> segs = Utility::split(str, ",");
			string key = segs[0];
			vector<string> key_segs = Utility::split(key, "\\");
			int label = subjs_dict[key_segs.at(0)];
			FaceClass fc(label, atoi(segs[2].c_str()), atoi(segs[3].c_str()), atoi(segs[4].c_str()), atoi(segs[5].c_str()));
			FrameFaceClass gt;
			gt.faceNum = 1;
			gt.faces.push_back(fc);
			gtMap.insert(make_pair(key, gt));

		}else{
			
			vector<string> segs = Utility::split(str, ";");
			string key = segs[0];
			int faceNum = atoi(segs[1].c_str());
			
			
			FrameFaceClass gt;
			gt.faceNum = faceNum;
			if (faceNum > 0){
				vector<string> faceStrings = Utility::split(segs[2], ":");
				for (int i = 0; i < faceNum; ++i){
					string faceString = faceStrings[i];
					vector<string> faceSegs = Utility::split(faceString, ",");
					int label = atoi(faceSegs.at(0).c_str());
					Rect face;
					face.x = atoi(faceSegs.at(1).c_str());
					face.y = atoi(faceSegs.at(2).c_str());
					face.width = atoi(faceSegs.at(3).c_str());
					face.height = atoi(faceSegs.at(4).c_str());
					FaceClass fc(label, face.x, face.y, face.width, face.height);
					gt.faces.push_back(fc);
				}
			}
			gtMap.insert(make_pair(key, gt));
		}
	}
	/**
	std::map < string, FrameFaceClass >::iterator iter;
	for (iter = gtMap.begin(); iter!= gtMap.end(); iter++){
		  cout << iter->first << endl;
		  cout << iter->second.faceNum << endl;
	}
	**/
	in.close();
}
void main(int argc, char *argv[])
{
	if (argc != 5){
		cout << "Usage:" << argv[0] << " subject_db_file frame_folder isYoutube output_annotation" << endl;
		exit(-1);
	}
	string output_annotation = argv[4];
	
	// choose mode

	if (output_annotation.compare("baseline") == 0){
		mode = JPEGCOMPRESS;
	}else if(output_annotation.compare("SimpleTracking") == 0){
		mode = SIMPLETRACKING;
	}else if(output_annotation.compare("FastForward") == 0){
		mode = FORWARDTRACKING;
	}else if(output_annotation.compare("FastForwardEWMA") == 0){
		mode = FORWARDTRACKINGEWMA;
	}else if(output_annotation.compare("SLIDE") == 0){
		mode = SLIDE;
	
	}else if(output_annotation.compare("baselineAgg") == 0){
			mode = HARIBASELINE;
	
	}
	//LTE wifi
	string outputFolder = "E:\\glimpse\\result\\featurePointTest_3g\\boundingBoxOnly\\data\\";
	// load string -> int subject map
	string subject_db_path = argv[1];
	subjs_dict = Utility::ReadSubjs(subject_db_path, labelToSubj);

	

	string isYoutbe = argv[3];
	if (atoi(isYoutbe.c_str()) == 1){
		isYoutubeDataset = true;
		frameRate = 24;
	}

	// init input video info
	std::string frame_folder = argv[2];		
	
	//C:\Users\yuhan\Glimpse\frame_images_DB\Aaron_Eckhart\0
	//C:\Users\yuhan\Glimpse\myDataset\WP_20131102_001
	// load groundtruth
	if (isYoutubeDataset){
		vector<string> folder_segs = Utility::split(frame_folder, "\\");
		string folder_name = folder_segs.at(folder_segs.size()-2);
		/**
		William_Overlin\0
		**/
		accuracyKey = folder_segs.at(folder_segs.size()-2) + "\\" + folder_segs.at(folder_segs.size()-1);
		loadGroundTruth("E:\\glimpse\\groundTruthFiles\\" + folder_name + ".labeled_faces.txt", isYoutubeDataset);
	}else{
		vector<string> folder_segs = Utility::split(frame_folder, "\\");
		string folder_name = folder_segs.at(folder_segs.size()-1);
		accuracyKey = folder_name;
		loadGroundTruth("E:\\glimpse\\groundTruthFilesWithCorrectRect\\" + folder_name + ".labeled_faces.txt", isYoutubeDataset);
	}
	
	
	// init socket
	NetworkService nwk(portNum, ipAddress);
	double min_latency = FLT_MAX;
	double delta;
	for (int i = 0 ; i < 5; ++i){
		double t1 = Utility::getCurrentTimeMSV2();
		double t2_server = nwk.sendSync();
		double t3 = Utility::getCurrentTimeMSV2();
		double latency = (t3 - t1)/2.0;
		if (latency < min_latency){
			min_latency = latency;
			delta = (t1 + min_latency) - t2_server;
		}
		Sleep(10);
	}
	printf("RTT: %10.6f", min_latency * 2);
	
	// init frame spitter
	FrameSpitter spitter(frame_folder, frameRate);

	// init motion analyzer
	MotionAnalyzer motionAnalyzer;

	// init logger
	string output_fileName;
	if (isYoutubeDataset){
		vector<string> fsegs = Utility::split(accuracyKey, "\\");
		output_fileName = fsegs[0]+ "_" + fsegs[1];
	}else{
		output_fileName = accuracyKey;
	}

	ofstream delay_fh;
	delay_fh.open(outputFolder  + output_annotation + "\\ "+ output_fileName + "_delay");
	ofstream accuracy_fh;
	accuracy_fh.open(outputFolder  + output_annotation + "\\ "+ output_fileName + "_accuracy");
	ofstream predictionLog_fh;
	predictionLog_fh.open(outputFolder  + output_annotation + "\\ "+ output_fileName  + "_prediction");
	ofstream gotoserver_fh;
	gotoserver_fh.open(outputFolder  + output_annotation + "\\ "+ output_fileName  + "_gotoserver");
	ofstream tracking_fh;
	tracking_fh.open(outputFolder  + output_annotation + "\\ "+ output_fileName  + "_tracking");

	// Start spitting frames
	double clock = 0;
	while(spitter.has_frame(clock)){

		++totalProcessedFrame;
		
		int frameNum;
		double captureTime = 0.0;
		string frameName;
		Mat curFrame = spitter.spit(clock, frameNum, captureTime, frameName);
		cout << "----------------start------------------" << endl << clock << "," << frameName << "," << frameNum << "," << captureTime << endl;
		if (curFrame.empty()){
			cout << "Error: Empty frame" << endl;
			exit(-1);
		}
		
		// Start processing the frame
		int transmittedByte = 0;	
		
		FrameFaceClass frameFace;
		
		double extraSendTime = 0;
		double redundantSleepingTime = 0.0;
		double devicePenaltyTime = 0.0;
		
		Rect motionRect;
		gotoServer = false;
		sentData = "";
		int operation;
		
		
		double t0 = Utility::getCurrentTimeMSV2();
		
		if (mode == CLIENTONLY){
			vector<Rect> faces = NativeDetectorWrapper((char *)curFrame.data,curFrame.rows, curFrame.cols);
		}
		else if (mode == HARIBASELINE){

			/** process response queue (from the past frames)**/
			vector<int> processedQID;
			vector<int> initNewFaceTrackerIndex;
			for(int q = queuedResponse.size()-1; q >= 0; --q){ /**TODO: might only need to process the latest response **/
				if (queuedResponse.at(q).processedTime < clock){ // process queued response
					for (int ahh = q; ahh >= 0; --ahh){
						processedQID.push_back(ahh);
					}
					cout  << "------------Response received!-------------" << endl;
					EntireFrameResponse response = queuedResponse.at(q);
					cout << "response:" << response.toString() <<  ",transmitTime:" << response.transmitTime << ",faceNum:" <<  response.faceNumber << endl;
					
					
					for (int i = 0; i < response.faceNumber; ++i){ // for each face in the response
						// adjusted rect and featurepoints will be stored in response -> and frameFace.
						FaceClass face = response.faces.at(i);
						Rect rectFromServer = face.faceRect;
						Rect updatedRect(rectFromServer.x + response.shift_x,  rectFromServer.y + response.shift_y, rectFromServer.width, rectFromServer.height);
						face.faceRect = updatedRect;

						for(int j = 0; j < face.featurePoints.size(); ++j){							
							face.featurePoints.at(j).x += response.shift_x;
							face.featurePoints.at(j).y += response.shift_y;
						}
						
						frameFace.faces.push_back(face);
			
					}// end of processing faces from a response
					
					break;
				}// end of processing all the responses which should be processed

			}// end of iterating through all the responses
			
			// remove processed response from the queue //
			for(int i = 0; i < processedQID.size() ; ++i){
				queuedResponse.erase(queuedResponse.begin() + processedQID.at(i));
			}
			/** end of processing response queue **/

			
			vector<unsigned char>buffer;
			Knobs::compressFrame(curFrame, COMPRESSLEVEL, buffer);
			// compression time expense, mean:39.18324655 std:11.37283646
			std::random_device rd;
			std::default_random_engine generator( rd() );
			std::normal_distribution<double> distribution(73.56380535, 10.50580556);
			devicePenaltyTime += distribution(generator);
					
			double tmp_t = Utility::getCurrentTimeMSV2();
			nwk.sendProcessFrameHeader(-1, buffer.size(), curFrame.cols, curFrame.rows, delta);
			Sleep(10);
			EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
			double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
			transmittedByte += buffer.size();

			response.theFrame = curFrame;
			response.transmitTime = clock;
			response.shift_x = motionRect.x;
			response.shift_y = motionRect.y;
			prevSentTime = clock;
					
			
			
			response.processedTime = clock + devicePenaltyTime + server_t - response.extraTime - 10;
			queuedResponse.push_back(response);
			extraSendTime += server_t;
			gotoServer = true;
			//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
			sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t)) + ";";
				
			cout << "---- Send Current Frame ---- " << endl <<" Will comeBack at:" << response.processedTime <<  endl;
			//cout << response.toString() << endl;
			
		}
		else if (mode == GRAYMOTIONSERVER){
			Mat croppedFrame;
			bool hasMotion = motionAnalyzer.isMoving(curFrame, motionRect);
			devicePenaltyTime += Utility::getMotionAnalysisTime();
			Knobs::cropFrame(curFrame, croppedFrame, motionRect.x, motionRect.y, motionRect.width, motionRect.height);
			/** grayscaling + motion analysis **/
			/**
			nwk.sendProcessFrameHeader(-1, croppedFrame.total() * croppedFrame.elemSize() , croppedFrame.cols, croppedFrame.rows, delta);
			Sleep(10);
			redundantSleepingTime += 10;
			EntireFrameResponse response = nwk.sendEntireFrame((char *)croppedFrame.data,   croppedFrame.total() * croppedFrame.elemSize());	
			transmittedByte += croppedFrame.total() * croppedFrame.elemSize();
			extraTime = response.extraTime;
			**/
			
			vector<unsigned char>buffer;
			Knobs::compressFrame(croppedFrame, COMPRESSLEVEL, buffer);
			devicePenaltyTime += Utility::getCompressionTime(motionRect.width, motionRect.height);
			
			nwk.sendProcessFrameHeader(-1, buffer.size(), motionRect.width, motionRect.height, delta);
			Sleep(10);
			redundantSleepingTime += 10;
			EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());	
			extraSendTime = response.extraTime;
			transmittedByte += buffer.size();
			
			for (int i = 0; i < response.faceNumber; ++i){
				FaceClass face = response.faces.at(i);
				Rect rectFromServer = face.faceRect;
				Rect updatedRect(rectFromServer.x + motionRect.x,  rectFromServer.y + motionRect.y, rectFromServer.width, rectFromServer.height);
				for(int j = 0; j < face.featurePoints.size(); ++j){							
					face.featurePoints.at(j).x += motionRect.x;
					face.featurePoints.at(j).y += motionRect.y;
				}

				face.faceRect = updatedRect;
				frameFace.faces.push_back(face);
			}
		}
		else if (mode == SERVERONLY){

			
			nwk.sendProcessFrameHeader(-1, curFrame.total() * curFrame.elemSize() , curFrame.cols, curFrame.rows, delta);
			Sleep(10);
			redundantSleepingTime += 10;
			EntireFrameResponse response = nwk.sendEntireFrame((char *)curFrame.data,   curFrame.total() * curFrame.elemSize());	
			transmittedByte += curFrame.total() * curFrame.elemSize();
			extraSendTime = response.extraTime;
			
			/**
			vector<unsigned char>buffer;
			Knobs::compressFrame(curFrame, COMPRESSLEVEL, buffer);
			// compression time expense, mean:39.18324655 std:11.37283646
			std::random_device rd;
			std::default_random_engine generator( rd() );
			std::normal_distribution<double> distribution(73.56380535, 10.50580556);
			devicePenaltyTime += distribution(generator);
			
			nwk.sendProcessFrameHeader(-1, buffer.size(), curFrame.cols, curFrame.rows, delta);
			Sleep(10);
			redundantSleepingTime += 10;
			EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());	
			extraTime = response.extraTime;
			transmittedByte += buffer.size();
			**/
			//cout << response.toString() << endl;
			
			for (int i = 0; i < response.faceNumber; ++i){
				frameFace.faces.push_back(response.faces.at(i));
			}
		}else if (mode == JPEGCOMPRESS){

			/** process response queue (from the past frames)**/
			vector<int> processedQID;
			vector<int> initNewFaceTrackerIndex;
			for(int q = queuedResponse.size()-1; q >= 0; --q){ /**TODO: might only need to process the latest response **/
				if (queuedResponse.at(q).processedTime < clock){ // process queued response
					processedQID.push_back(q);
					cout  << "------------Response received!-------------" << endl;
					EntireFrameResponse response = queuedResponse.at(q);
					cout << "response:" << response.toString() <<  ",transmitTime:" << response.transmitTime << ",faceNum:" <<  response.faceNumber << endl;
					
					
					for (int i = 0; i < response.faceNumber; ++i){ // for each face in the response
						// adjusted rect and featurepoints will be stored in response -> and frameFace.
						FaceClass face = response.faces.at(i);
						Rect rectFromServer = face.faceRect;
						Rect updatedRect(rectFromServer.x + response.shift_x,  rectFromServer.y + response.shift_y, rectFromServer.width, rectFromServer.height);
						face.faceRect = updatedRect;

						for(int j = 0; j < face.featurePoints.size(); ++j){							
							face.featurePoints.at(j).x += response.shift_x;
							face.featurePoints.at(j).y += response.shift_y;
						}
						
						frameFace.faces.push_back(face);
			
					}// end of processing faces from a response
					
				}// end of processing all the responses which should be processed

			}// end of iterating through all the responses
			
			// remove processed response from the queue //
			for(int i = processedQID.size()-1; i >= 0; --i){
				queuedResponse.erase(queuedResponse.begin() + processedQID.at(i));
			}
			/** end of processing response queue **/

			if (queuedResponse.size() == 0){
				trackingState = 0;
			}	


			bool doMotionAnalysis = false;
			/** send delta frame **/
			if (trackingState == 0){ 

				if (doMotionAnalysis){
				// detect moving region //
					Mat croppedFrame;
					bool hasMotion = motionAnalyzer.isMoving(curFrame, motionRect);
					devicePenaltyTime +=  Utility::getMotionAnalysisTime();

					Knobs::cropFrame(curFrame, croppedFrame, motionRect.x, motionRect.y, motionRect.width, motionRect.height);
							
					// Send compressed frame
					vector<unsigned char>buffer;
					Knobs::compressFrame(croppedFrame, COMPRESSLEVEL, buffer);
					devicePenaltyTime += Utility::getCompressionTime(motionRect.width, motionRect.height);
					
					double tmp_t = Utility::getCurrentTimeMSV2();
					nwk.sendProcessFrameHeader(-1, buffer.size(), croppedFrame.cols, croppedFrame.rows, delta);
					Sleep(10);
					EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
					double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
					transmittedByte += buffer.size();			
					
					response.theFrame = curFrame;
					response.transmitTime = clock;
					response.shift_x = motionRect.x;
					response.shift_y = motionRect.y;
					prevSentTime = clock;
					
					response.processedTime = clock + devicePenaltyTime + server_t - response.extraTime - 10;
					queuedResponse.push_back(response);
					// need to be deducted from current processing time //
					// plus delta_t cuase it's async //
					extraSendTime += (response.extraTime + 10);
					gotoServer = true;
					//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
					sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-extraSendTime+devicePenaltyTime)) + ";";
				
					cout << "---- Send Current Frame ---- " << endl <<" Will comeBack at:" << response.processedTime <<  endl;
					//cout << response.toString() << endl;
				}else{// send compressed frame
					vector<unsigned char>buffer;
					Knobs::compressFrame(curFrame, COMPRESSLEVEL, buffer);
					// compression time expense, mean:39.18324655 std:11.37283646
					std::random_device rd;
					std::default_random_engine generator( rd() );
					std::normal_distribution<double> distribution(73.56380535, 10.50580556);
					devicePenaltyTime += distribution(generator);
					
					double tmp_t = Utility::getCurrentTimeMSV2();
					nwk.sendProcessFrameHeader(-1, buffer.size(), curFrame.cols, curFrame.rows, delta);
					Sleep(10);
					EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
					double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
					transmittedByte += buffer.size();

					response.theFrame = curFrame;
					response.transmitTime = clock;
					response.shift_x = motionRect.x;
					response.shift_y = motionRect.y;
					prevSentTime = clock;
					
					response.processedTime = clock + devicePenaltyTime + server_t - response.extraTime - 10;
					queuedResponse.push_back(response);
					extraSendTime += (response.extraTime + 10);
					gotoServer = true;
					//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
					sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-extraSendTime+devicePenaltyTime)) + ";";
				
					cout << "---- Send Current Frame ---- " << endl <<" Will comeBack at:" << response.processedTime <<  endl;
					//cout << response.toString() << endl;
			
				}
				trackingState = 1;
			}
			/** end of sending delta frame **/
			
			
		
		}else if (mode == SIMPLETRACKING){
			
			// only useful when no motion
			if (clock > nextSentTime && queuedResponse.size() == 0 && trackingState == 1){
				trackingState = 0;
			}	

			/** process response queue (from the past frames)**/
			vector<int> processedQID;
			vector<int> initNewFaceTrackerIndex;
			for(int q = queuedResponse.size()-1; q >= 0; --q){ /**TODO: might only need to process the latest response **/
				if (queuedResponse.at(q).processedTime < clock){ // process queued response
					trackingState = 0;
					processedQID.push_back(q);
					trackers.clear();
					cout  << "------------Response received!-------------" << endl;
					EntireFrameResponse response = queuedResponse.at(q);
					cout << "response:" << response.toString() <<  ",transmitTime:" << response.transmitTime << ",faceNum:" <<  response.faceNumber << endl;
							
					for (int i = 0; i < response.faceNumber; ++i){ // for each face in the response
						// adjusted rect and featurepoints will be stored in response -> and frameFace.
						FaceClass face = response.faces.at(i);
						Rect rectFromServer = face.faceRect;
						Rect updatedRect(rectFromServer.x + response.shift_x,  rectFromServer.y + response.shift_y, rectFromServer.width, rectFromServer.height);
						face.faceRect = updatedRect;

						for(int j = 0; j < face.featurePoints.size(); ++j){							
							face.featurePoints.at(j).x += response.shift_x;
							face.featurePoints.at(j).y += response.shift_y;
						}
						
						// merge trackers
						bool goodTrackerFound = false;
						bool shouldInitNewTracker = true;
						/**
						vector<int> removedTracker;
						for (int j = trackers.size()-1; j >= 0; --j){ // for each currently tracked face
							Rect trackRect = trackers.at(j).fc.faceRect;
							Rect curRect = face.faceRect;
							if (distance_main(trackRect.x, trackRect.y, curRect.x, curRect.y) < 7000 || face.label == trackers.at(j).fc.label){
											
								
								//tracking the same face, trust the one with higher confidence
								
								if(trackers.at(j).fc.confidence > face.confidence){
									
									// use tracker if the confidence from detection is low
									shouldInitNewTracker = false;
									goodTrackerFound = true;

								}else{
									// use detection result, remove bad trackers
									cout << "merging:" << face.label << endl;
									removedTracker.push_back(j);				
								}

							}
						}
						// 
						for (int j = 0; j < removedTracker.size(); ++j){	
							cout << "remove:" << trackers.at(removedTracker.at(j)).fc.label << endl;
							trackers.erase(trackers.begin()+removedTracker.at(j));
						}
						**/
						if (shouldInitNewTracker){
							FaceTracker tracker;
							tracker.init(face, response.theFrame);
							trackers.push_back(tracker);
						}
						
					}// end of processing faces from a response
					
				}// end of processing all the responses which should be processed

			}// end of iterating through all the responses
			for(int i = processedQID.size()-1; i >= 0; --i){
				queuedResponse.erase(queuedResponse.begin() + processedQID.at(i));
			}
			/** end of processing response queue **/



			/** send delta frame **/
			if (trackingState == 0){ 
				bool doMotionAnalysis = false;

				// detect moving region //
				if (doMotionAnalysis){
					Mat croppedFrame;
					bool hasMotion = motionAnalyzer.isMoving(curFrame, motionRect);
					devicePenaltyTime +=  Utility::getMotionAnalysisTime();

					if (hasMotion){ 
						
						Knobs::cropFrame(curFrame, croppedFrame, motionRect.x, motionRect.y, motionRect.width, motionRect.height);
							
						// Send compressed frame
						vector<unsigned char>buffer;
						Knobs::compressFrame(croppedFrame, COMPRESSLEVEL, buffer);
						devicePenaltyTime += Utility::getCompressionTime(motionRect.width, motionRect.height);
					
						double tmp_t = Utility::getCurrentTimeMSV2();
						nwk.sendProcessFrameHeader(-1, buffer.size(), croppedFrame.cols, croppedFrame.rows, delta);
						Sleep(10);
						EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
						double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
						transmittedByte += buffer.size();			
					
						response.theFrame = curFrame;
						response.transmitTime = clock;
						response.shift_x = motionRect.x;
						response.shift_y = motionRect.y;
						prevSentTime = clock;
					
						response.processedTime = clock +  server_t + devicePenaltyTime - response.extraTime - 10;
						queuedResponse.push_back(response);
						// need to be deducted from current processing time //
						// plus delta_t cuase it's async //
						extraSendTime += (server_t);
						gotoServer = true;
						//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
						sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-10-response.extraTime+devicePenaltyTime)) + ";";
						
						cout << "---- Sent ---- " << endl <<" Will comeBack at:" << response.processedTime << endl;
					}
					double d_ave = 0.0;
					for (int di = 0; di < aveDelay.size(); ++di){
						d_ave += aveDelay.at(di);
					}

					nextSentTime = clock + (d_ave / (aveDelay.size() *1.0 )) ;
					
				
				}else{
					vector<unsigned char>buffer;
					Knobs::compressFrame(curFrame, COMPRESSLEVEL, buffer);
					// compression time expense, mean:39.18324655 std:11.37283646
					std::random_device rd;
					std::default_random_engine generator( rd() );
					std::normal_distribution<double> distribution(73.56380535, 10.50580556);
					devicePenaltyTime += distribution(generator);
					
					double tmp_t = Utility::getCurrentTimeMSV2();
					nwk.sendProcessFrameHeader(-1, buffer.size(), curFrame.cols, curFrame.rows, delta);
					Sleep(10);
					EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
					double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
					transmittedByte += buffer.size();

					response.theFrame = curFrame;
					response.transmitTime = clock;
					response.shift_x = 0;
					response.shift_y = 0;
					prevSentTime = clock;
					
					response.processedTime = clock + devicePenaltyTime + server_t - response.extraTime - 10;
					queuedResponse.push_back(response);
					extraSendTime += (server_t);
					gotoServer = true;
					//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
					sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-10-response.extraTime+devicePenaltyTime)) + ";";
						
					cout << "---- Send Current Frame ---- " << endl <<" Will comeBack at:" << response.processedTime <<  endl;
					//cout << response.toString() << endl;
				}
				trackingState = 1;
			}
			/** end of sending delta frame **/
			
			/** tracking **/
			cout << "----- Start tracking -----" << endl;
			cout << "tracker size: " << trackers.size() << endl;
			int trackResult;
			for (int i = trackers.size() - 1; i >= 0; --i){	
				Rect prevRect = trackers.at(i).fc.faceRect;
			
				// Track face
				vector<Point2f> featurePoints;
				Rect trackedFaceRect;
				double tmp_t = Utility::getCurrentTimeMSV2();
				double estimatedTrackTime;
				trackResult = trackers.at(i).trackWholeFrame(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				//trackResult = trackers.at(i).subRegionTrack(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				//trackResult = trackers.at(i).track(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime); // drift
				
				cout << "track " <<  trackers.at(i).fc.label << ": time:" << estimatedTrackTime << endl;
				devicePenaltyTime += estimatedTrackTime;
				
				if (trackResult == -1){//tracking failed
					//trackers.clear();
					trackers.erase(trackers.begin() + i);
					//frameFace.faces.clear();
					//trackingState = 0;
					break;

				}else{ // tracking works, rectangle in trackedFaceRect	
					cout << "tracking works" << endl;
					frameFace.faces.push_back(trackers.at(i).fc);
					
				}

			}
			/** end of tracking**/
			
	}else if (mode == SLIDE){
			// only useful when no motion
			if (clock > nextSentTime && queuedResponse.size() == 0 && trackingState == 1){
				trackingState = 0;
			}	


			/** process response queue (from the past frames)**/
			vector<int> processedQID;
			vector<int> initNewFaceTrackerIndex;
			for(int q = queuedResponse.size()-1; q >= 0; --q){ /**TODO: might only need to process the latest response **/
				if (queuedResponse.at(q).processedTime < clock){ // process queued response
					processedQID.push_back(q);
					trackers.clear();
					
					trackingState = 0;
					
					cout  << "------------Response received!-------------" << endl;
					EntireFrameResponse response = queuedResponse.at(q);
					cout << "response:" << response.toString() <<  ",transmitTime:" << response.transmitTime << ",faceNum:" <<  response.faceNumber << endl;
					
					for (int i = 0; i < response.faceNumber; ++i){ // for each face in the response
						// adjusted rect and featurepoints will be stored in response -> and frameFace.
						FaceClass face = response.faces.at(i);
						Rect rectFromServer = face.faceRect;
						Rect updatedRect(rectFromServer.x + response.shift_x,  rectFromServer.y + response.shift_y, rectFromServer.width, rectFromServer.height);
						face.faceRect = updatedRect;

						for(int j = 0; j < face.featurePoints.size(); ++j){							
							face.featurePoints.at(j).x += response.shift_x;
							face.featurePoints.at(j).y += response.shift_y;
						}
						
						// slide //
						
						FaceTracker tracker;
						tracker.init(face, response.theFrame);
						bool successful = true;
							
						vector<Point2f> featurePoints;
						Rect trackedFaceRect;
						for (int cachedFIndex = 0; cachedFIndex < cachedFrames.size(); ++cachedFIndex){
								
							double estimatedTrackTime;
							int replayTrackResult = tracker.trackWholeFrame(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
								
							//int replayTrackResult = tracker.track(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
							
							devicePenaltyTime += estimatedTrackTime;
							//cout << "!!!!replay:"  << cachedFrames.at(cachedFIndex).frameName  << ". estimated time:" << estimatedTrackTime << endl;
							if ( replayTrackResult < 0){
								successful = false;
								break;
							}
						}
							
						if (successful){
							trackers.push_back(tracker);
						}else{
							cout << "reaply failed" << endl;
						}
						

						// merge trackers // 
						// be smarter here!! //
						/**
						bool goodTrackerFound = false;
						bool shouldInitNewTracker = true;
						**/
						/** reuse detection result**/
						/**
						vector<int> removedTracker;
						for (int j = trackers.size()-1; j >= 0; --j){ // for each currently tracked face
							Rect trackRect = trackers.at(j).fc.faceRect;
							Rect curRect = face.faceRect;
							if (distance_main(trackRect.x , trackRect.y , curRect.x , curRect.y) < 9000 || Utility::overLap(trackRect, curRect) == 1 || face.label == trackers.at(j).fc.label){
								
								if (face.label != trackers.at(j).fc.label){
									if(trackers.at(j).fc.confidence > face.confidence){
									
										// use tracker if the confidence from detection is low
										shouldInitNewTracker = false;
										goodTrackerFound = true;

									}
								}else{ // face label == face label 
									// use detection result, remove bad trackers
									cout << "merging:" << face.label << endl;
									removedTracker.push_back(j);				
								}
							}
						}
						// 
						for (int j = 0; j < removedTracker.size(); ++j){	
							cout << "remove:" << trackers.at(removedTracker.at(j)).fc.label << endl;
							trackers.erase(trackers.begin()+removedTracker.at(j));
						}
							**/
						/**
						if (shouldInitNewTracker){
						
							
						}**/
						
					}// end of processing faces from a response
					
				}// end of processing all the responses which should be processed

			}// end of iterating through all the responses

			for(int i = processedQID.size()-1; i >= 0; --i){
				queuedResponse.erase(queuedResponse.begin() + processedQID.at(i));
			}
			/** end of processing response queue **/


			
			/** send delta frame **/
			if (trackingState == 0){ 
				cachedFrames.clear();
				bool doMotionAnalysis = false;

				// detect moving region //
				if (doMotionAnalysis){
					Mat croppedFrame;
					bool hasMotion = motionAnalyzer.isMoving(curFrame, motionRect);
					devicePenaltyTime +=  Utility::getMotionAnalysisTime();

					if (hasMotion){ 
						
						Knobs::cropFrame(curFrame, croppedFrame, motionRect.x, motionRect.y, motionRect.width, motionRect.height);
							
						// Send compressed frame
						vector<unsigned char>buffer;
						Knobs::compressFrame(croppedFrame, COMPRESSLEVEL, buffer);
						devicePenaltyTime += Utility::getCompressionTime(motionRect.width, motionRect.height);
					
						double tmp_t = Utility::getCurrentTimeMSV2();
						nwk.sendProcessFrameHeader(-1, buffer.size(), croppedFrame.cols, croppedFrame.rows, delta);
						Sleep(10);
						EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
						double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
						transmittedByte += buffer.size();			
					
						response.theFrame = curFrame;
						response.transmitTime = clock;
						response.shift_x = motionRect.x;
						response.shift_y = motionRect.y;
						prevSentTime = clock;
					
						response.processedTime = clock +  server_t + devicePenaltyTime - response.extraTime - 10;
						queuedResponse.push_back(response);
						// need to be deducted from current processing time //
						// plus delta_t cuase it's async //
						extraSendTime += (server_t);
						gotoServer = true;
						//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
						sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-10-response.extraTime+devicePenaltyTime)) + ";";
						
						cout << "---- Sent ---- " << endl <<" Will comeBack at:" << response.processedTime << endl;
					}
					double d_ave = 0.0;
					for (int di = 0; di < aveDelay.size(); ++di){
						d_ave += aveDelay.at(di);
					}

					nextSentTime = clock + (d_ave / (aveDelay.size() *1.0 )) ;
					
				
				}else{
					vector<unsigned char>buffer;
					Knobs::compressFrame(curFrame, COMPRESSLEVEL, buffer);
					// compression time expense, mean:39.18324655 std:11.37283646
					std::random_device rd;
					std::default_random_engine generator( rd() );
					std::normal_distribution<double> distribution(73.56380535, 10.50580556);
					devicePenaltyTime += distribution(generator);
					
					double tmp_t = Utility::getCurrentTimeMSV2();
					nwk.sendProcessFrameHeader(-1, buffer.size(), curFrame.cols, curFrame.rows, delta);
					Sleep(10);
					EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
					double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
					transmittedByte += buffer.size();

					response.theFrame = curFrame;
					response.transmitTime = clock;
					response.shift_x = 0;
					response.shift_y = 0;
					prevSentTime = clock;
					
					response.processedTime = clock + devicePenaltyTime + server_t - response.extraTime - 10;
					queuedResponse.push_back(response);
					extraSendTime += (server_t);
					gotoServer = true;
					//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
					sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-10-response.extraTime+devicePenaltyTime)) + ";";
						
					cout << "---- Send Current Frame ---- " << endl <<" Will comeBack at:" << response.processedTime <<  endl;
					//cout << response.toString() << endl;
				}
				trackingState = 1;
			}
			/** end of sending delta frame **/
			
		
			/** tracking **/
			int trackResult;
			for (int i = trackers.size() - 1; i >= 0; --i){	
				Rect prevRect = trackers.at(i).fc.faceRect;
			
				// Track face
				cout << "start normal tracking..." << endl;
				vector<Point2f> featurePoints;
				Rect trackedFaceRect;
				double tmp_t = Utility::getCurrentTimeMSV2();
				double estimatedTrackTime;
				//trackResult = trackers.at(i).trackWholeFrame(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				//trackResult = trackers.at(i).subRegionTrack(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				trackResult = trackers.at(i).track(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);

				
				cout << "normal tracking...:" << estimatedTrackTime << endl;
				devicePenaltyTime += estimatedTrackTime;
				
				if (trackResult == -1){//tracking failed
					//trackers.clear();
					trackers.erase(trackers.begin() + i);
					//frameFace.faces.clear();
					//trackingState = 0;
					break;

				}else{ // tracking works, rectangle in trackedFaceRect	
					cout << "tracking works" << endl;
					frameFace.faces.push_back(trackers.at(i).fc);
					
				}

			}
			/** end of tracking**/
			cout << "end of tracking..." << endl;
			
			/** Caching intermediate frames **/
			if (queuedResponse.size() > 0){
				CachedFrame interFrame(curFrame, clock, frameName);
				cachedFrames.push_back(interFrame);
				//cout << "cache frame: " <<  frameName << endl;
			}
			
			
	}
	else if (mode == FORWARDTRACKING){
			// only useful when no motion
			if (clock > nextSentTime && queuedResponse.size() == 0 && trackingState == 1){
				trackingState = 0;
			}	


			/** process response queue (from the past frames)**/
			vector<int> processedQID;
			vector<int> initNewFaceTrackerIndex;
			for(int q = queuedResponse.size()-1; q >= 0; --q){ /**TODO: might only need to process the latest response **/
				if (queuedResponse.at(q).processedTime < clock){ // process queued response
					processedQID.push_back(q);
					trackers.clear();
					trackingState = 0;
					
					cout  << "------------Response received!-------------" << endl;
					EntireFrameResponse response = queuedResponse.at(q);
					cout << "response:" << response.toString() <<  ",transmitTime:" << response.transmitTime << ",faceNum:" <<  response.faceNumber << endl;
					
					for (int i = 0; i < response.faceNumber; ++i){ // for each face in the response
						// adjusted rect and featurepoints will be stored in response -> and frameFace.
						FaceClass face = response.faces.at(i);
						Rect rectFromServer = face.faceRect;
						Rect updatedRect(rectFromServer.x + response.shift_x,  rectFromServer.y + response.shift_y, rectFromServer.width, rectFromServer.height);
						face.faceRect = updatedRect;

						for(int j = 0; j < face.featurePoints.size(); ++j){							
							face.featurePoints.at(j).x += response.shift_x;
							face.featurePoints.at(j).y += response.shift_y;
						}
						
						// slide //
						/**
						FaceTracker tracker;
						tracker.init(face, response.theFrame);
						bool successful = true;
							
						vector<Point2f> featurePoints;
						Rect trackedFaceRect;
						for (int cachedFIndex = 0; cachedFIndex < cachedFrames.size(); ++cachedFIndex){
								
							double estimatedTrackTime;
							double tmp_t = Utility::getCurrentTimeMSV2();
							//tracker.trackWholeFrame
							int replayTrackResult = tracker.trackWholeFrame(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
								
							//int replayTrackResult = tracker.track(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
							double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
							devicePenaltyTime += estimatedTrackTime - delta_t;
							//cout << "!!!!replay:"  << cachedFrames.at(cachedFIndex).frameName  << ". estimated time:" << estimatedTrackTime << endl;
							if ( replayTrackResult < 0){
								successful = false;
								break;
							}
						}
							
						if (successful){
							trackers.push_back(tracker);
						}else{
							cout << "reaply failed" << endl;
						}
						**/	

							
							// Adaptive fast forward //
							
							FaceTracker tracker;
							tracker.init(face, response.theFrame);
							bool successful = true;
							
							vector<Point2f> featurePoints;
							Rect trackedFaceRect;
							int cachedFIndex = 0;
							int prevCachedFIndex = 0;
							while( cachedFIndex < cachedFrames.size()){
								
								double estimatedTrackTime;
								double tmp_t = Utility::getCurrentTimeMSV2();
								int replayTrackResult = tracker.track(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
								double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
								devicePenaltyTime += estimatedTrackTime - delta_t;
								cout << "<replay> "  << cachedFrames.at(cachedFIndex).frameName  << ". estimated time:" << estimatedTrackTime << endl;
							
								if ( replayTrackResult < 0){ //if at any time the tracker fails, the game is over.
									successful = false;
									break;
								}

								if (cachedFIndex < 1){ // no history
									prevCachedFIndex = cachedFIndex;
									++cachedFIndex;
								}else{ // use history to predict next processed frames
									Rect curRect = tracker.faceTrajectroy.at(tracker.faceTrajectroy.size()-1);
									Rect prevRect = tracker.faceTrajectroy.at(tracker.faceTrajectroy.size()-2);
									double deltaTimeMs = (cachedFrames.at(cachedFIndex).timeStamp - cachedFrames.at(prevCachedFIndex).timeStamp);
									cout << "cachedFIndex: " << cachedFIndex << ", prevCachedIndex: " << prevCachedFIndex << ", t2:" <<cachedFrames.at(cachedFIndex).timeStamp 
										<<",t1: " << cachedFrames.at(prevCachedFIndex).timeStamp << endl;
									double dist = Utility::myDistance(curRect.x, curRect.y, prevRect.x, prevRect.y);
									float velocityMs = dist/(deltaTimeMs*1.0);
									
									// determine next cachedFIndex
									double skipTime = cachedFrames.at(cachedFIndex).timeStamp + (7.0/velocityMs);
									cout << "deltaTimeMs:" << deltaTimeMs << ",dist:"  << dist << ", velocityMs:" <<  velocityMs  << ", skiptTime:"  << skipTime;

									prevCachedFIndex = cachedFIndex;
									int fi;
									for (fi = cachedFIndex+1 ; fi < cachedFrames.size(); ++fi){
										if (cachedFrames.at(fi).timeStamp >= skipTime){
											cachedFIndex = fi;
											break;
										}
									}
									if (fi == cachedFrames.size()){
										break;
									}
									//cout << "velocity" << velocity << ",nextIndex:" << cachedFIndex << endl;
								}
							
							}
							
							if (successful){
								/**
								vector<int> removedTracker;
								for (int j = trackers.size()-1; j >= 0; --j){ // for each currently tracked face
									Rect trackRect = trackers.at(j).fc.faceRect;
									Rect curRect = face.faceRect;
									if (distance_main(trackRect.x, trackRect.y, curRect.x, curRect.y) < 7000 || face.label == trackers.at(j).fc.label ){
										//tracking the same face, trust the one with higher confidence
										removedTracker.push_back(j);				
									}
								}
						 
								for (int j = 0; j < removedTracker.size(); ++j){	
									trackers.erase(trackers.begin()+removedTracker.at(j));
								}
								**/
								trackers.push_back(tracker);
							}
							
							


						// merge trackers // 
						// be smarter here!! //
						/**
						bool goodTrackerFound = false;
						bool shouldInitNewTracker = true;
						**/
						/** reuse detection result**/
						/**
						vector<int> removedTracker;
						for (int j = trackers.size()-1; j >= 0; --j){ // for each currently tracked face
							Rect trackRect = trackers.at(j).fc.faceRect;
							Rect curRect = face.faceRect;
							if (distance_main(trackRect.x , trackRect.y , curRect.x , curRect.y) < 9000 || Utility::overLap(trackRect, curRect) == 1 || face.label == trackers.at(j).fc.label){
								
								if (face.label != trackers.at(j).fc.label){
									if(trackers.at(j).fc.confidence > face.confidence){
									
										// use tracker if the confidence from detection is low
										shouldInitNewTracker = false;
										goodTrackerFound = true;

									}
								}else{ // face label == face label 
									// use detection result, remove bad trackers
									cout << "merging:" << face.label << endl;
									removedTracker.push_back(j);				
								}
							}
						}
						// 
						for (int j = 0; j < removedTracker.size(); ++j){	
							cout << "remove:" << trackers.at(removedTracker.at(j)).fc.label << endl;
							trackers.erase(trackers.begin()+removedTracker.at(j));
						}
							**/
						/**
						if (shouldInitNewTracker){
						
							
						}**/
						
					}// end of processing faces from a response
					
				}// end of processing all the responses which should be processed

			}// end of iterating through all the responses

			for(int i = processedQID.size()-1; i >= 0; --i){
				queuedResponse.erase(queuedResponse.begin() + processedQID.at(i));
			}
			/** end of processing response queue **/


		
			/** send delta frame **/
			if (trackingState == 0){ 
				cachedFrames.clear();
				bool doMotionAnalysis = false;

				// detect moving region //
				if (doMotionAnalysis){
					Mat croppedFrame;
					bool hasMotion = motionAnalyzer.isMoving(curFrame, motionRect);
					devicePenaltyTime +=  Utility::getMotionAnalysisTime();

					if (hasMotion){ 
						
						Knobs::cropFrame(curFrame, croppedFrame, motionRect.x, motionRect.y, motionRect.width, motionRect.height);
							
						// Send compressed frame
						vector<unsigned char>buffer;
						Knobs::compressFrame(croppedFrame, COMPRESSLEVEL, buffer);
						devicePenaltyTime += Utility::getCompressionTime(motionRect.width, motionRect.height);
					
						double tmp_t = Utility::getCurrentTimeMSV2();
						nwk.sendProcessFrameHeader(-1, buffer.size(), croppedFrame.cols, croppedFrame.rows, delta);
						Sleep(10);
						EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
						double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
						transmittedByte += buffer.size();			
					
						response.theFrame = curFrame;
						response.transmitTime = clock;
						response.shift_x = motionRect.x;
						response.shift_y = motionRect.y;
						prevSentTime = clock;
					
						response.processedTime = clock +  server_t + devicePenaltyTime - response.extraTime - 10;
						queuedResponse.push_back(response);
						// need to be deducted from current processing time //
						// plus delta_t cuase it's async //
						extraSendTime += (server_t);
						gotoServer = true;
						//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
						sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-10-response.extraTime+devicePenaltyTime)) + ";";
						
						cout << "---- Sent ---- " << endl <<" Will comeBack at:" << response.processedTime << endl;
					}
					double d_ave = 0.0;
					for (int di = 0; di < aveDelay.size(); ++di){
						d_ave += aveDelay.at(di);
					}

					nextSentTime = clock + (d_ave / (aveDelay.size() *1.0 )) ;
					
				
				}else{
					vector<unsigned char>buffer;
					Knobs::compressFrame(curFrame, COMPRESSLEVEL, buffer);
					// compression time expense, mean:39.18324655 std:11.37283646
					std::random_device rd;
					std::default_random_engine generator( rd() );
					std::normal_distribution<double> distribution(73.56380535, 10.50580556);
					devicePenaltyTime += distribution(generator);
					
					double tmp_t = Utility::getCurrentTimeMSV2();
					nwk.sendProcessFrameHeader(-1, buffer.size(), curFrame.cols, curFrame.rows, delta);
					Sleep(10);
					EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
					double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
					transmittedByte += buffer.size();

					response.theFrame = curFrame;
					response.transmitTime = clock;
					response.shift_x = 0;
					response.shift_y = 0;
					prevSentTime = clock;
					
					response.processedTime = clock + devicePenaltyTime + server_t - response.extraTime - 10;
					queuedResponse.push_back(response);
					extraSendTime += (server_t);
					gotoServer = true;
					//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
					sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-10-response.extraTime+devicePenaltyTime)) + ";";
						
					cout << "---- Send Current Frame ---- " << endl <<" Will comeBack at:" << response.processedTime <<  endl;
					//cout << response.toString() << endl;
				}
				trackingState = 1;
			}
			/** end of sending delta frame **/
			
		
			/** tracking **/
			int trackResult;
			for (int i = trackers.size() - 1; i >= 0; --i){	
				Rect prevRect = trackers.at(i).fc.faceRect;
			
				// Track face
				cout << "start normal tracking..." << endl;
				vector<Point2f> featurePoints;
				Rect trackedFaceRect;
				double tmp_t = Utility::getCurrentTimeMSV2();
				double estimatedTrackTime;
				//trackResult = trackers.at(i).trackWholeFrame(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				//trackResult = trackers.at(i).subRegionTrack(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				trackResult = trackers.at(i).track(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);

				
				cout << "normal tracking...:" << estimatedTrackTime << endl;
				devicePenaltyTime += estimatedTrackTime;
				
				if (trackResult == -1){//tracking failed
					//trackers.clear();
					trackers.erase(trackers.begin() + i);
					//frameFace.faces.clear();
					//trackingState = 0;
					break;

				}else{ // tracking works, rectangle in trackedFaceRect	
					cout << "tracking works" << endl;
					frameFace.faces.push_back(trackers.at(i).fc);
					
				}

			}
			/** end of tracking**/
			cout << "end of tracking..." << endl;
			
			/** Caching intermediate frames **/
			if (queuedResponse.size() > 0){
				CachedFrame interFrame(curFrame, clock, frameName);
				cachedFrames.push_back(interFrame);
				//cout << "cache frame: " <<  frameName << endl;
			}
			
			
	}else if (mode == FORWARDTRACKINGEWMA){
			// only useful when no motion
			if (clock > nextSentTime && queuedResponse.size() == 0 && trackingState == 1){
				trackingState = 0;
			}	


			/** process response queue (from the past frames)**/
			vector<int> processedQID;
			vector<int> initNewFaceTrackerIndex;
			vector<int> hitTrackers;
			int oriTrackerSize = trackers.size();
			for(int q = queuedResponse.size()-1; q >= 0; --q){ /**TODO: might only need to process the latest response **/
				if (queuedResponse.at(q).processedTime < clock){ // process queued response
					processedQID.push_back(q);
					trackingState = 0;
					
					//cout  << "------------Response received!-------------" << endl;
					EntireFrameResponse response = queuedResponse.at(q);
					//cout << "response:" << response.toString() <<  ",transmitTime:" << response.transmitTime << ",faceNum:" <<  response.faceNumber << endl;
					
					for (int i = 0; i < response.faceNumber; ++i){ // for each face in the response
						// adjusted rect and featurepoints will be stored in response -> and frameFace.
						FaceClass face = response.faces.at(i);
						Rect rectFromServer = face.faceRect;
						Rect updatedRect(rectFromServer.x + response.shift_x,  rectFromServer.y + response.shift_y, rectFromServer.width, rectFromServer.height);
						face.faceRect = updatedRect;

						for(int j = 0; j < face.featurePoints.size(); ++j){							
							face.featurePoints.at(j).x += response.shift_x;
							face.featurePoints.at(j).y += response.shift_y;
						}
						
					
							// Adaptive fast forward //
							
							FaceTracker tracker;
							tracker.init(face, response.theFrame);
							bool successful = true;
							
							vector<Point2f> featurePoints;
							Rect trackedFaceRect;
							int cachedFIndex = 0;
							int prevCachedFIndex = 0;
							while( cachedFIndex < cachedFrames.size()){
								
								double estimatedTrackTime;
								double tmp_t = Utility::getCurrentTimeMSV2();
								int replayTrackResult = tracker.track(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
								double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
								devicePenaltyTime += estimatedTrackTime - delta_t;
								//cout << "<replay> "  << cachedFrames.at(cachedFIndex).frameName  << ". estimated time:" << estimatedTrackTime << endl;
							
								if ( replayTrackResult < 0){ //if at any time the tracker fails, the game is over.
									successful = false;
									break;
								}

								if (cachedFIndex < 1){ // no history
									prevCachedFIndex = cachedFIndex;
									++cachedFIndex;
								}else{ // use history to predict next processed frames
									Rect curRect = tracker.faceTrajectroy.at(tracker.faceTrajectroy.size()-1);
									Rect prevRect = tracker.faceTrajectroy.at(tracker.faceTrajectroy.size()-2);
									double deltaTimeMs = (cachedFrames.at(cachedFIndex).timeStamp - cachedFrames.at(prevCachedFIndex).timeStamp);
									//cout << "cachedFIndex: " << cachedFIndex << ", prevCachedIndex: " << prevCachedFIndex << ", t2:" <<cachedFrames.at(cachedFIndex).timeStamp 
									//	<<",t1: " << cachedFrames.at(prevCachedFIndex).timeStamp << endl;
									double dist = Utility::myDistance(curRect.x, curRect.y, prevRect.x, prevRect.y);
									float velocityMs = dist/(deltaTimeMs*1.0);
									
									// determine next cachedFIndex
									double skipTime = cachedFrames.at(cachedFIndex).timeStamp + (7.0/velocityMs);
									//cout << "deltaTimeMs:" << deltaTimeMs << ",dist:"  << dist << ", velocityMs:" <<  velocityMs  << ", skiptTime:"  << skipTime;

									prevCachedFIndex = cachedFIndex;
									int fi;
									for (fi = cachedFIndex+1 ; fi < cachedFrames.size(); ++fi){
										if (cachedFrames.at(fi).timeStamp >= skipTime){
											cachedFIndex = fi;
											break;
										}
									}
									if (fi == cachedFrames.size()){
										break;
									}
									//cout << "velocity" << velocity << ",nextIndex:" << cachedFIndex << endl;
								}
							
							}
							
							if (successful){
								
								
								bool hit = false;
								for (int j = 0; j < oriTrackerSize; ++j){ // for each currently tracked face
									Rect trackRect = trackers.at(j).fc.faceRect;
									Rect curRect = tracker.fc.faceRect;
									if (distance_main(trackRect.x + trackRect.width/2, trackRect.y + trackRect.height/2, curRect.x + curRect.width/2, curRect.y + curRect.height/2) < 900 ||
										Utility::commonAuBPortion(trackRect, curRect) > 0.2
										|| Utility::overLap(trackRect, tracker.fc.faceRect) == 1){
										//tracking the same face, trust the one with higher confidence
										//removedTracker.push_back(j);
										hitTrackers.push_back(j);
										hit = true;
										int max_index = 0;
										double max_value = 0.0;
										for (int k = 0; k < tracker.fc.ewma.size(); ++k){
											tracker.fc.ewma.at(k) = trackers.at(j).fc.ewma.at(k) * 0.6 + tracker.fc.ewma.at(k) * 0.4;
											if (tracker.fc.ewma.at(k) > max_value){
												max_value = tracker.fc.ewma.at(k);
												max_index = k;
											}
										}
										tracker.fc.label = max_index;
										tracker.fc.confidence = max_value;
										trackers.at(j) = tracker;
									}
									
								}
						 
								if (!hit){
									trackers.push_back(tracker);
								}
							}
						
					}// end of processing faces from a response

					// update trackers
				for (int i = 0; i < oriTrackerSize; ++i){
					bool hit = false;
					for (int j = 0; j < hitTrackers.size(); ++j){
						if (i == hitTrackers.at(j)){
							hit = true;
							break;
						}
					}
					if (!hit){
						int max_index = 0;
						double max_value = 0.0;
						for (int k = 0; k < trackers.at(i).fc.ewma.size(); ++k){
							trackers.at(i).fc.ewma.at(k) = trackers.at(i).fc.ewma.at(k) * 0.6 + 0 * 0.4;
							if (trackers.at(i).fc.ewma.at(k) > max_value){
								max_value = trackers.at(i).fc.ewma.at(k);
								max_index = k;
							}
						}
						trackers.at(i).fc.confidence = max_value;
						trackers.at(i).fc.label = max_index;
					}
			}

			for (int i = trackers.size() - 1; i >= 0; --i){
				if (trackers.at(i).fc.confidence < 0.3){
					trackers.erase(trackers.begin() + i);
				}
			}

					
				}// end of processing all the responses which should be processed

			}// end of iterating through all the responses

			for(int i = processedQID.size()-1; i >= 0; --i){
				queuedResponse.erase(queuedResponse.begin() + processedQID.at(i));
			}
			/** end of processing response queue **/

			
			
			/** send delta frame **/
			if (trackingState == 0){ 
				cachedFrames.clear();
				bool doMotionAnalysis = false;

				// detect moving region //
				if (doMotionAnalysis){
					Mat croppedFrame;
					bool hasMotion = motionAnalyzer.isMoving(curFrame, motionRect);
					devicePenaltyTime +=  Utility::getMotionAnalysisTime();

					if (hasMotion){ 
						
						Knobs::cropFrame(curFrame, croppedFrame, motionRect.x, motionRect.y, motionRect.width, motionRect.height);
							
						// Send compressed frame
						vector<unsigned char>buffer;
						Knobs::compressFrame(croppedFrame, COMPRESSLEVEL, buffer);
						devicePenaltyTime += Utility::getCompressionTime(motionRect.width, motionRect.height);
					
						double tmp_t = Utility::getCurrentTimeMSV2();
						nwk.sendProcessFrameHeader(-1, buffer.size(), croppedFrame.cols, croppedFrame.rows, delta);
						Sleep(10);
						EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
						double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
						transmittedByte += buffer.size();			
					
						response.theFrame = curFrame;
						response.transmitTime = clock;
						response.shift_x = motionRect.x;
						response.shift_y = motionRect.y;
						prevSentTime = clock;
					
						response.processedTime = clock +  server_t + devicePenaltyTime - response.extraTime - 10;
						queuedResponse.push_back(response);
						// need to be deducted from current processing time //
						// plus delta_t cuase it's async //
						extraSendTime += (server_t);
						gotoServer = true;
						//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
						sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-10-response.extraTime+devicePenaltyTime)) + ";";
						
						cout << "---- Sent ---- " << endl <<" Will comeBack at:" << response.processedTime << endl;
					}
					double d_ave = 0.0;
					for (int di = 0; di < aveDelay.size(); ++di){
						d_ave += aveDelay.at(di);
					}

					nextSentTime = clock + (d_ave / (aveDelay.size() *1.0 )) ;
					
				
				}else{
					vector<unsigned char>buffer;
					Knobs::compressFrame(curFrame, COMPRESSLEVEL, buffer);
					// compression time expense, mean:39.18324655 std:11.37283646
					std::random_device rd;
					std::default_random_engine generator( rd() );
					std::normal_distribution<double> distribution(73.56380535, 10.50580556);
					devicePenaltyTime += distribution(generator);
					
					double tmp_t = Utility::getCurrentTimeMSV2();
					nwk.sendProcessFrameHeader(-1, buffer.size(), curFrame.cols, curFrame.rows, delta);
					Sleep(10);
					EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
					double server_t = Utility::getCurrentTimeMSV2() - tmp_t;
					transmittedByte += buffer.size();

					response.theFrame = curFrame;
					response.transmitTime = clock;
					response.shift_x = 0;
					response.shift_y = 0;
					prevSentTime = clock;
					
					response.processedTime = clock + devicePenaltyTime + server_t - response.extraTime - 10;
					queuedResponse.push_back(response);
					extraSendTime += (server_t);
					gotoServer = true;
					//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
					sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((server_t-10-response.extraTime+devicePenaltyTime)) + ";";
						
					cout << "---- Send Current Frame ---- " << endl <<" Will comeBack at:" << response.processedTime <<  endl;
					//cout << response.toString() << endl;
				}
				trackingState = 1;
			}
			/** end of sending delta frame **/
		
			/** tracking **/
			int trackResult;
			for (int i = trackers.size() - 1; i >= 0; --i){	
				Rect prevRect = trackers.at(i).fc.faceRect;
			
				// Track face
				
				vector<Point2f> featurePoints;
				Rect trackedFaceRect;
				double tmp_t = Utility::getCurrentTimeMSV2();
				double estimatedTrackTime;
				//trackResult = trackers.at(i).trackWholeFrame(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				//trackResult = trackers.at(i).subRegionTrack(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				trackResult = trackers.at(i).track(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);

				
				//cout << "normal tracking...:" << estimatedTrackTime << endl;
				devicePenaltyTime += estimatedTrackTime;
				
				if (trackResult == -1){//tracking failed
					//trackers.clear();
					trackers.erase(trackers.begin() + i);
					//frameFace.faces.clear();
					//trackingState = 0;
					break;

				}

			}
			/** end of tracking**/
			//cout << "end of tracking..." << endl;
			
			/** Caching intermediate frames **/
			if (queuedResponse.size() > 0){
				CachedFrame interFrame(curFrame, clock, frameName);
				cachedFrames.push_back(interFrame);
				//cout << "cache frame: " <<  frameName << endl;
			}
			
			// post process
			vector<int> badFaces;
			for (int i = 0; i < trackers.size(); ++i){
				for (int j = i+1; j < trackers.size(); ++j){
					if (trackers.at(i).fc.label == trackers.at(j).fc.label){
						if (trackers.at(i).fc.confidence > trackers.at(j).fc.confidence){
							badFaces.push_back(j);
						}else{
							badFaces.push_back(i);
						
						}
					}
				}
			}
			/**
			for (int i = 0; i < trackers.size(); ++i){

				if (trackers.at(i).fc.faceRect.width < 40 ||trackers.at(i).fc.faceRect.height < 40 ){
					bool hit = false;
					for (int j = 0; j < badFaces.size(); ++j){
						if ( i == badFaces.at(j)){
							hit = true;
							break;
						}
					}
					if (!hit){
						badFaces.push_back(i);
					}

				}

			}**/
			std::sort(badFaces.begin(), badFaces.end());
			// remove duplicate faces
			for (int i = badFaces.size() - 1; i >= 0; --i){
				trackers.erase(trackers.begin() + badFaces.at(i));
			}
			
			//copy to frameFace
			for (int i = 0; i <trackers.size(); ++i){
				frameFace.faces.push_back(trackers.at(i).fc);
			}
	}


	else if (mode == IMAGEREGISTRATION){
			
			/** send delta frame **/
			if (trackingState == 0){ 

				// detect moving region //
				Mat croppedFrame;
				double tmp_t = Utility::getCurrentTimeMSV2();
				bool hasMotion = motionAnalyzer.isMoving(curFrame, motionRect);
				double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
				devicePenaltyTime +=  Utility::getMotionAnalysisTime() - delta_t;
			
					/**
					if ((motionRect.width * motionRect.height - pixCounter) < 1/2.0){
						hasMotion = false;
					}
					**/
				if (hasMotion){ 

					//trackedRegion
					/**
					Mat tmpFrame;
					curFrame.copyTo(tmpFrame);
					int pixCounter = 0;

					for(int yi = trackedRegion.y ; yi < (trackedRegion.y + trackedRegion.height) ; ++yi){
						for(int xi = trackedRegion.x ; xi < trackedRegion.width + trackedRegion.x; ++xi){
							int index =  yi * tmpFrame.cols + xi;
							int inside = false;
							//cout << "tracker size:" << trackers.size() << endl;
							
							if (trackers.size() > 0){
								for (int trackerIndex = 0 ; trackerIndex <  (trackers.size()- 1); ++trackerIndex){
								//	cout << "fi:" << trackerIndex << ",tracker size:" << trackers.size() << endl;
									if ((clock - trackers.at(trackerIndex).startTime) > 1000 && xi > trackers.at(trackerIndex).fc.faceRect.x && xi < (trackers.at(trackerIndex).fc.faceRect.x + trackers.at(trackerIndex).fc.faceRect.width)
										&& yi > trackers.at(trackerIndex).fc.faceRect.y  && yi < (trackers.at(trackerIndex).fc.faceRect.y + trackers.at(trackerIndex).fc.faceRect.height)){
										// do nothing
											++pixCounter;
											inside = true;
											break;
									}
								}
							}
							if (!inside){
									tmpFrame.data[index] = 0;
							}
							
						}
					}
					imshow("tmpFrame", tmpFrame);
					waitKey(33);
					**/
					Knobs::cropFrame(curFrame, croppedFrame, motionRect.x, motionRect.y, motionRect.width, motionRect.height);
					

					// Send compressed frame
					vector<unsigned char>buffer;
					tmp_t = Utility::getCurrentTimeMSV2();
					Knobs::compressFrame(croppedFrame, COMPRESSLEVEL, buffer);
					delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
					devicePenaltyTime += Utility::getCompressionTime(motionRect.width, motionRect.height) - delta_t;
					
					tmp_t = Utility::getCurrentTimeMSV2();
					nwk.sendProcessFrameHeader(-1, buffer.size(), croppedFrame.cols, croppedFrame.rows, delta);
					Sleep(10);
					EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
					delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
					transmittedByte += buffer.size();			
					
					response.theFrame = curFrame;
					response.transmitTime = clock;
					response.shift_x = motionRect.x;
					response.shift_y = motionRect.y;
					prevSentTime = clock;
					
					response.processedTime = clock +  delta_t + devicePenaltyTime - response.extraTime - 10;
					queuedResponse.push_back(response);
					// need to be deducted from current processing time //
					// plus delta_t cuase it's async //
					extraSendTime += (response.extraTime + 10 + delta_t);
					gotoServer = true;
					//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
					sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((delta_t-10-response.extraTime+devicePenaltyTime)) + ";";
					cachedFrames.clear();
					cout << "---- Sent ---- " << endl <<" will comeBack at:" << response.processedTime << endl;
					//cout << response.toString() << endl;
				}
				trackingState = 1;
			}
			/** end of sending delta frame **/
			
			/** process response queue (from the past frames)**/
			int processFacesNum = 0;
			vector<int> processedQID;
			vector<int> initNewFaceTrackerIndex;
			for(int q = queuedResponse.size()-1; q >= 0; --q){ /**TODO: might only need to process the latest response **/
				if (queuedResponse.at(q).processedTime < clock){ // process queued response
					processedQID.push_back(q);
					//trackers.clear();
					cout << endl << "------------Response received!-------------" << endl;
					EntireFrameResponse response = queuedResponse.at(q);
					cout << "response:" << response.toString() << "faceNum:" <<  response.faceNumber << endl;
					
					processFacesNum += response.faceNumber;
					for (int i = 0; i < response.faceNumber; ++i){ // for each face in the response
						// adjusted rect and featurepoints will be stored in response -> and frameFace.
						FaceClass face = response.faces.at(i);
						Rect rectFromServer = face.faceRect;
						Rect updatedRect(rectFromServer.x + response.shift_x,  rectFromServer.y + response.shift_y, rectFromServer.width, rectFromServer.height);
						face.faceRect = updatedRect;

						for(int j = 0; j < face.featurePoints.size(); ++j){							
							face.featurePoints.at(j).x += response.shift_x;
							face.featurePoints.at(j).y += response.shift_y;
						}
						
						// merge trackers
						bool goodTrackerFound = false;
						bool shouldInitNewTracker = true;
						/** use new result **/
						
						vector<int> removedTracker;
						for (int j = trackers.size()-1; j >= 0; --j){ // for each currently tracked face
							Rect trackRect = trackers.at(j).fc.faceRect;
							Rect curRect = face.faceRect;
							if (distance_main(trackRect.x, trackRect.y, curRect.x, curRect.y) < 7000 || face.label == trackers.at(j).fc.label ){
								//tracking the same face, trust the one with higher confidence
									removedTracker.push_back(j);				
							}
						}
						// 
						for (int j = 0; j < removedTracker.size(); ++j){	
							cout << "remove:" << trackers.at(removedTracker.at(j)).fc.label << endl;
							trackers.erase(trackers.begin()+removedTracker.at(j));
						}
						
						/** reuse detection result**/
						/**
						vector<int> removedTracker;
						for (int j = trackers.size()-1; j >= 0; --j){ // for each currently tracked face
							Rect trackRect = trackers.at(j).fc.faceRect;
							Rect curRect = face.faceRect;
							if (distance_main(trackRect.x, trackRect.y, curRect.x, curRect.y) < 7500 || face.label == trackers.at(j).fc.label){
								//tracking the same face, trust the one with higher confidence
								if(trackers.at(j).fc.confidence > face.confidence){
									
									// use tracker if the confidence from detection is low
									shouldInitNewTracker = false;
									goodTrackerFound = true;

								}else{
									// use detection result, remove bad trackers
									cout << "merging:" << face.label << endl;
									removedTracker.push_back(j);				
								}
							}
						}
						// 
						for (int j = 0; j < removedTracker.size(); ++j){	
							cout << "remove:" << trackers.at(removedTracker.at(j)).fc.label << endl;
							trackers.erase(trackers.begin()+removedTracker.at(j));
						}**/
							
						
						if (shouldInitNewTracker){
							/**
							FaceTracker tracker;
							tracker.init(face, response.theFrame);
							bool successful = true;
							
							vector<Point2f> featurePoints;
							Rect trackedFaceRect;
							for (int cachedFIndex = 0; cachedFIndex < cachedFrames.size(); ++cachedFIndex){
								
								int estimatedTrackTime;
								double tmp_t = Utility::getCurrentTimeMSV2();
								int replayTrackResult = tracker.track(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
								double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
								devicePenaltyTime += estimatedTrackTime - delta_t;
								//cout << "!!!!replay:"  << cachedFrames.at(cachedFIndex).frameName  << ". estimated time:" << estimatedTrackTime << endl;
								if ( replayTrackResult < 0){
									successful = false;
									break;
								}
							}
							
							if (successful){
								trackers.push_back(tracker);
							}else{
								cout << "reaply failed" << endl;
							}
							**/
							// Adaptive fast forward //
							
							FaceTracker tracker;
							tracker.init(face, response.theFrame, clock);
							bool successful = true;
							
							vector<Point2f> featurePoints;
							Rect trackedFaceRect;
							int cachedFIndex = 0;
							int prevCachedFIndex = 0;
							while( cachedFIndex < cachedFrames.size()){
								
								double estimatedTrackTime;
								double tmp_t = Utility::getCurrentTimeMSV2();
								int replayTrackResult = tracker.track(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
								double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
								devicePenaltyTime += estimatedTrackTime - delta_t;
								cout << "!!!!replay:"  << cachedFrames.at(cachedFIndex).frameName  << ". estimated time:" << estimatedTrackTime << endl;
								// Dmax = 7;
								//for (int fi = 0; fi < featurePoints.size(); ++fi){
								//	circle( cachedFrames.at(cachedFIndex).frame, featurePoints[fi], 3, cv::Scalar(0,255,0), -1 , 8);
								//}
								//cv::imshow("croppedFrame", croppedPrevFrame);
								//cv::waitKey( 50 );
								
								//imshow("reply", cachedFrames.at(cachedFIndex).frame);
								//waitKey(30);
								if ( replayTrackResult < 0){ //if at any time the tracker fails, the game is over.
									successful = false;
									break;
								}

								if (cachedFIndex < 1){ // no history
									prevCachedFIndex = cachedFIndex;
									++cachedFIndex;
								}else{ // use history to predict next processed frames
									Rect curRect = tracker.faceTrajectroy.at(tracker.faceTrajectroy.size()-1);
									Rect prevRect = tracker.faceTrajectroy.at(tracker.faceTrajectroy.size()-2);
									double deltaTimeMs = (cachedFrames.at(cachedFIndex).timeStamp - cachedFrames.at(prevCachedFIndex).timeStamp);
									cout << "cachedFIndex: " << cachedFIndex << ", prevCachedIndex: " << prevCachedFIndex << ", t2:" <<cachedFrames.at(cachedFIndex).timeStamp 
										<<",t1: " << cachedFrames.at(prevCachedFIndex).timeStamp << endl;
									double dist = Utility::myDistance(curRect.x, curRect.y, prevRect.x, prevRect.y);
									float velocityMs = dist/(deltaTimeMs*1.0);
									
									// determine next cachedFIndex
									double skipTime = cachedFrames.at(cachedFIndex).timeStamp + (7.0/velocityMs);
									cout << "deltaTimeMs:" << deltaTimeMs << ",dist:"  << dist << ", velocityMs:" <<  velocityMs  << ", skiptTime:"  << skipTime;

									prevCachedFIndex = cachedFIndex;
									int fi;
									for (fi = cachedFIndex+1 ; fi < cachedFrames.size(); ++fi){
										if (cachedFrames.at(fi).timeStamp >= skipTime){
											cachedFIndex = fi;
											break;
										}
									}
									if (fi == cachedFrames.size()){
										break;
									}
									//cout << "velocity" << velocity << ",nextIndex:" << cachedFIndex << endl;
								}
							
							}
							
							if (successful){
								trackers.push_back(tracker);
							}
							

							
						}
						
					}// end of processing faces from a response
					
				}// end of processing all the responses which should be processed

			}// end of iterating through all the responses
			
			cout << "before removing queue size:" <<queuedResponse.size() << endl;
			for(int i = processedQID.size()-1; i >= 0; --i){
				queuedResponse.erase(queuedResponse.begin() + processedQID.at(i));
			}
			cout << "after removing queue size:" << queuedResponse.size() << endl;
			/** end of processing response queue **/

			/** tracking **/
			cout << "tracker size: " << trackers.size() << endl;
			int trackResult;
			for (int i = trackers.size() - 1; i >= 0; --i){	
				Rect prevRect = trackers.at(i).fc.faceRect;
			
				// Track face
				cout << "start normal tracking..." << endl;
				vector<Point2f> featurePoints;
				Rect trackedFaceRect;
				double tmp_t = Utility::getCurrentTimeMSV2();
				double estimatedTrackTime;
				//trackResult = trackers.at(i).trackWholeFrame(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				//trackResult = trackers.at(i).subRegionTrack(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				trackResult = trackers.at(i).track(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);

				double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
				cout << "normal tracking...:" << estimatedTrackTime << endl;
				devicePenaltyTime += estimatedTrackTime - delta_t;
				
				if (trackResult == -1){//tracking failed
					//trackers.clear();
					trackers.erase(trackers.begin() + i);
					//frameFace.faces.clear();
					//trackingState = 0;
					break;

				}else{ // tracking works, rectangle in trackedFaceRect	
					cout << "tracking works" << endl;
					frameFace.faces.push_back(trackers.at(i).fc);
					
				}

			}
			/** end of tracking**/
			cout << "end of tracking..." << endl;
			
			/** Caching intermediate frames **/
			if (queuedResponse.size() > 0){
				CachedFrame interFrame(curFrame, clock, frameName);
				cachedFrames.push_back(interFrame);
				//cout << "cache frame: " <<  frameName << endl;
			}
			/** Done with caching **/
			if (queuedResponse.size() == 0){
				trackingState = 0;
			}	

			
			/** decide if we need to send the next frame to the server 
			** should be smarter **/

			if (clock > nextWholeFrameProcessingTime){
				if (prevProcessedFrame.cols == 0 || prevProcessedFrame.rows == 0){
					curFrame.copyTo(prevProcessedFrame);
					processedRegionTracker.init(prevProcessedFrame);
				}else{
					int estimatedTrackingTime = 0;
					curFrame.copyTo(prevProcessedFrame);
					processedRegionTracker.track(curFrame, estimatedTrackingTime);
					nextWholeFrameProcessingTime += estimatedTrackingTime;
					// check if we need to do motion analysis

					if (processedRegionTracker.isLost(trackedRegion) && queuedResponse.size() == 0){ //yes, we need to do motion analysis
						
						prevProcessedFrame.release();
						trackingState = 0;
					}/**
					Scalar color = colors[4%8];
					rectangle(curFrame, cvPoint(trackedRegion.x, trackedRegion.y), cvPoint((trackedRegion.x + trackedRegion.width), trackedRegion.y + trackedRegion.height),
					color, 3, 8, 0);
					cout <<"trackedReegion:" << trackedRegion.x << ","<< trackedRegion.y << "," << trackedRegion.x + trackedRegion.width << "," << trackedRegion.y + trackedRegion.height << endl;
					imshow("trackedRegion", curFrame);
					waitKey(60);
					**/
				}
			}
			
	}
		/******* yuhan ********/
		else if (mode == CONTINUETRACKING){
			
			/** send delta frame **/
			if (trackingState == 0){ 

				// detect moving region //
				Mat croppedFrame;
				double tmp_t = Utility::getCurrentTimeMSV2();
				bool hasMotion = motionAnalyzer.isMoving(curFrame, motionRect);
				double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
				devicePenaltyTime += 36 - delta_t;
				cout << "motionRect b4: " << motionRect.x << "," << motionRect.y << "," << motionRect.x + motionRect.width << "," << motionRect.y + motionRect.height <<endl;
					
				if (hasMotion && !(trackedRegion.x <= motionRect.x && (trackedRegion.x + trackedRegion.width) >= (motionRect.x + motionRect.width)
						&& trackedRegion.y <= motionRect.y && (trackedRegion.y + trackedRegion.height) >= (motionRect.y + motionRect.height))){
				//if (hasMotion){ 
						

					// image registration //
					//areas with motion - trackedRegion 
					// list all four possibilities
					if (trackedRegion.x <= motionRect.x && (trackedRegion.x + trackedRegion.width) >= (motionRect.x + motionRect.width))
					{
						if (trackedRegion.y <= motionRect.y && (trackedRegion.y + trackedRegion.height) <= (motionRect.y + motionRect.height)){
							motionRect.y = trackedRegion.y + trackedRegion.height;
							motionRect.height = (motionRect.x + motionRect.height) - (trackedRegion.x + trackedRegion.height);
							}
						if (trackedRegion.y >= motionRect.y && (trackedRegion.y + trackedRegion.height) >= (motionRect.y + motionRect.height)){
							motionRect.height = trackedRegion.y - motionRect.y;
						}
					}else if (trackedRegion.y <= motionRect.y && (trackedRegion.y + trackedRegion.height) >= (motionRect.y + motionRect.height)){
						if (trackedRegion.x <= motionRect.x && (trackedRegion.x + trackedRegion.width) <= (motionRect.x + motionRect.width)){
							motionRect.x = trackedRegion.x + trackedRegion.width;
							motionRect.width = (motionRect.x + motionRect.width) - (trackedRegion.x + trackedRegion.width);
						}
						if (trackedRegion.x >= motionRect.x && (trackedRegion.x + trackedRegion.width) >= (motionRect.x + motionRect.width)){
							motionRect.width = trackedRegion.x - motionRect.x;
						}
					}
					
					operation = 0;
					cout << "Normal operation with motion" << endl;
					cout << "motionRect after: " << motionRect.x << "," << motionRect.y << "," << motionRect.x + motionRect.width << "," << motionRect.y + motionRect.height <<endl;
					cout << "trackedRegion: " << trackedRegion.x << "," << trackedRegion.y << "," << trackedRegion.x + trackedRegion.width << "," << trackedRegion.y + trackedRegion.height <<endl;
					
					Knobs::cropFrame(curFrame, croppedFrame, motionRect.x, motionRect.y, motionRect.width, motionRect.height);
							
					
					// Send compressed frame
					vector<unsigned char>buffer;
					tmp_t = Utility::getCurrentTimeMSV2();
					Knobs::compressFrame(croppedFrame, COMPRESSLEVEL, buffer);
					delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
					devicePenaltyTime += 35 - delta_t;
					
					tmp_t = Utility::getCurrentTimeMSV2();
					nwk.sendProcessFrameHeader(-1, buffer.size(), croppedFrame.cols, croppedFrame.rows, delta);
					Sleep(10);
					EntireFrameResponse response = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
					delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
					transmittedByte += buffer.size();			
					
					
					
					// Send the frame without compression
					/**
					tmp_t = Utility::getCurrentTimeMSV2();
					nwk.sendProcessFrameHeader(-1, croppedFrame.total() * croppedFrame.elemSize(), croppedFrame.cols, croppedFrame.rows, delta);
					Sleep(10);
					EntireFrameResponse response = nwk.sendEntireFrame((char *)croppedFrame.data, croppedFrame.total() * croppedFrame.elemSize());
					transmittedByte += croppedFrame.total() * croppedFrame.elemSize();
					delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
					**/

					response.theFrame = curFrame;
					response.transmitTime = clock;
					response.shift_x = motionRect.x;
					response.shift_y = motionRect.y;
					prevSentTime = clock;
					
					
					response.processedTime = clock +  delta_t + devicePenaltyTime - response.extraTime - 10;
					queuedResponse.push_back(response);
					// need to be deducted from current processing time //
					extraSendTime += (response.extraTime + 10 + delta_t);
					gotoServer = true;
					//sentData = frameName + ",0," + to_string(croppedFrame.total() * croppedFrame.elemSize()) + "," + to_string((delta_t-10-response.extraTime)) + ";";
					sentData = frameName + ",0," + to_string(buffer.size()) + "," + to_string((delta_t-10-response.extraTime+devicePenaltyTime)) + ";";
					
					cout << "---- Sent ---- " << endl <<" will comeBack at:" << response.processedTime << endl;
					cachedFrames.clear();
					//cout << response.toString() << endl;
				}
				trackingState = 1;
			}
			/** end of sending delta frame **/
			
			/** process response queue (from the past frames)**/
			int processFacesNum = 0;
			vector<int> processedQID;
			vector<int> initNewFaceTrackerIndex;
			for(int q = queuedResponse.size()-1; q >= 0; --q){ /**TODO: might only need to process the latest response **/
				if (queuedResponse.at(q).processedTime < clock){ // process queued response
					processedQID.push_back(q);
					
					cout << endl << "------------Response received!-------------" << endl;
					EntireFrameResponse response = queuedResponse.at(q);
					cout << "response:" << response.toString() << "faceNum:" <<  response.faceNumber << endl;
					
					processFacesNum += response.faceNumber;
					for (int i = 0; i < response.faceNumber; ++i){ // for each face in the response
						// adjusted rect and featurepoints will be stored in response -> and frameFace.
						FaceClass face = response.faces.at(i);
						Rect rectFromServer = face.faceRect;
						Rect updatedRect(rectFromServer.x + response.shift_x,  rectFromServer.y + response.shift_y, rectFromServer.width, rectFromServer.height);
						face.faceRect = updatedRect;

						for(int j = 0; j < face.featurePoints.size(); ++j){							
							face.featurePoints.at(j).x += response.shift_x;
							face.featurePoints.at(j).y += response.shift_y;
						}
						
						// merge trackers
						bool goodTrackerFound = false;
						bool shouldInitNewTracker = true;
						vector<int> removedTracker;
						for (int j = trackers.size()-1; j >= 0; --j){ // for each currently tracked face
							Rect trackRect = trackers.at(j).fc.faceRect;
							Rect curRect = face.faceRect;
							if (distance_main(trackRect.x, trackRect.y, curRect.x, curRect.y) < 7000 || face.label == trackers.at(j).fc.label){
								//tracking the same face, trust the one with higher confidence
								if(trackers.at(j).fc.confidence > face.confidence){
									
									// use tracker if the confidence from detection is low
									shouldInitNewTracker = false;
									goodTrackerFound = true;

								}else{

									// use detection result, remove bad trackers
									cout << "merging:" << face.label << endl;
									removedTracker.push_back(j);				
								}
							}
						}
						// 
						for (int j = 0; j < removedTracker.size(); ++j){	
							cout << "remove:" << trackers.at(removedTracker.at(j)).fc.label << endl;
							trackers.erase(trackers.begin()+removedTracker.at(j));
						}
						
						if (shouldInitNewTracker){
							
							FaceTracker tracker;
							tracker.init(face, response.theFrame);
							bool successful = true;
							
							vector<Point2f> featurePoints;
							Rect trackedFaceRect;
							for (int cachedFIndex = 0; cachedFIndex < cachedFrames.size(); ++cachedFIndex){
								
								double estimatedTrackTime;
								double tmp_t = Utility::getCurrentTimeMSV2();
								int replayTrackResult = tracker.track(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
								double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
								devicePenaltyTime += estimatedTrackTime - delta_t;
								cout << "!!!!replay:"  << cachedFrames.at(cachedFIndex).frameName  << ". estimated time:" << estimatedTrackTime << endl;
								if ( replayTrackResult < 0){
									
									successful = false;
									break;
								}else{
									for(int fID = 0; fID < featurePoints.size(); ++fID){
										circle(cachedFrames.at(cachedFIndex).frame, featurePoints[fID], 3, cv::Scalar(0,255,0), -1 , 8);
									}
								}
							}
							
							if (successful){
								trackers.push_back(tracker);
							}
							

							// Adaptive fast forward //
							/**
							FaceTracker tracker;
							tracker.init(face, response.theFrame);
							bool successful = true;
							
							vector<Point2f> featurePoints;
							Rect trackedFaceRect;
							int cachedFIndex = 0;
							while( cachedFIndex < cachedFrames.size()){
								
								int estimatedTrackTime;
								double tmp_t = Utility::getCurrentTimeMSV2();
								int replayTrackResult = tracker.track(cachedFrames.at(cachedFIndex).frame, featurePoints, trackedFaceRect, estimatedTrackTime);
								double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
								devicePenaltyTime += estimatedTrackTime - delta_t;
								cout << "!!!!replay:"  << cachedFrames.at(cachedFIndex).frameName  << ". estimated time:" << estimatedTrackTime << endl;
								// Dmax = 7;

								if ( replayTrackResult < 0){ //if at any time the tracker fails, the game is over.
									successful = false;
									break;
								}

								if (cachedFIndex < 1){ // no history
									++cachedFIndex;
								}else{ // use history to predict next processed frames
									Rect curRect = tracker.faceTrajectroy.at(tracker.faceTrajectroy.size()-1);
									Rect prevRect = tracker.faceTrajectroy.at(tracker.faceTrajectroy.size()-2);
									float velocity = Utility::myDistance(curRect.x, curRect.y, prevRect.x, prevRect.y);

									cachedFIndex += (7.0/velocity)/frameRate;
								}
								
							}
							
							if (successful){
								trackers.push_back(tracker);
							}
							**/
						}
						
					}// end of processing faces from a response
					
				}// end of processing all the responses which should be processed

			}// end of iterating through all the responses
			
			/**
			for (int i = 0; i < trackers.size(); ++i){
				frameFace.faces.push_back(trackers.at(i).fc);
			}
			**/
			cout << "before removing queue size:" <<queuedResponse.size() << endl;
			for(int i = processedQID.size()-1; i >= 0; --i){
				queuedResponse.erase(queuedResponse.begin() + processedQID.at(i));
			}
			cout << "after removing queue size:" << queuedResponse.size() << endl;
			/** end of processing response queue **/

			/** tracking **/
			cout << "tracker size: " << trackers.size() << endl;
			int trackResult;
			//for (int i = 0; i < trackers.size(); ++i){
			for (int i = trackers.size() - 1; i >= 0; --i){
				
				Rect prevRect = trackers.at(i).fc.faceRect;
			
				// Track face
				cout << "start normal tracking..." << endl;
				vector<Point2f> featurePoints;
				Rect trackedFaceRect;
				double tmp_t = Utility::getCurrentTimeMSV2();
				double estimatedTrackTime;
				trackResult = trackers.at(i).track(curFrame, featurePoints, trackedFaceRect, estimatedTrackTime);
				double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
				cout << "normal tracking...:" << estimatedTrackTime << endl;
				devicePenaltyTime += estimatedTrackTime - delta_t;
				
				if (trackResult == -1 || trackResult == -2){//tracking failed
					operation = 2;
					//trackers.clear();
					trackers.erase(trackers.begin() + i);
					//frameFace.faces.clear();
					//trackingState = 0;
					break;

				}else{ // tracking works, rectangle in trackedFaceRect	
					cout << "tracking works" << endl;
					frameFace.faces.push_back(trackers.at(i).fc);
					/**
					if (tracker.fc.wantMore == 1){
						operation = 4;
						cout << "tracking works, server wants more" << endl;	
						// enlarge the rectangle a bit
						Mat croppedFace;
						Rect enlargedRect;
						Utility::enlargeROI(curFrame, trackedFaceRect, croppedFace, enlargedRect);
						
						// compress				
						vector<unsigned char>buffer;
						double tmp_t = Utility::getCurrentTimeMSV2();
						Knobs::compressFrame(croppedFace, COMPRESSLEVEL, buffer);
						double delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
						devicePenaltyTime += (30 - delta_t);
								
						// send
							
						tmp_t = Utility::getCurrentTimeMSV2();
						nwk.sendProcessFrameHeader(tracker.fc.faceID, (int)buffer.size(), croppedFace.cols, croppedFace.rows, delta);
						Sleep(10);
						EntireFrameResponse singleFaceResponse = nwk.sendEntireFrame((char *)buffer.data(), buffer.size());
						transmittedByte += buffer.size();
							
								
						tmp_t = Utility::getCurrentTimeMSV2();
						nwk.sendProcessFrameHeader(-1, croppedFace.total() * croppedFace.elemSize(), croppedFace.cols, croppedFace.rows, delta);
						Sleep(10);
						EntireFrameResponse singleFaceResponse = nwk.sendEntireFrame((char *)croppedFace.data ,croppedFace.total() * croppedFace.elemSize());
						transmittedByte += croppedFace.total() * croppedFace.elemSize();			
							
						delta_t = Utility::getCurrentTimeMSV2() - tmp_t;
						singleFaceResponse.theFrame = curFrame;
						singleFaceResponse.transmitTime = clock;
						singleFaceResponse.shift_x = enlargedRect.x;
						singleFaceResponse.shift_y = enlargedRect.y;
							
						singleFaceResponse.processedTime = clock +  delta_t - singleFaceResponse.extraTime -10;
						cout << "comeBack:" << singleFaceResponse.processedTime << endl;
						queuedResponse.push_back(singleFaceResponse);
						extraSendTime += delta_t;
						gotoServer = true;
						sentData += frameName + ",1," + to_string(croppedFace.total() * croppedFace.elemSize()) + "," + to_string((delta_t-10-singleFaceResponse.extraTime)) + ";";
				
	
					
						//N1 = singleFaceResponse.N1;
						//serverExecTime = singleFaceResponse.serverExecutionTime;
						//N2 = singleFaceResponse.N2;
								
								
						
					}else{ // no need to send more
						operation = 6;
						cout << "tracking works, no need to send more" << endl;
						frameFace.faces.push_back(tracker.fc);
					}
					**/
				}

			}
			/** end of tracking**/

			cout << "end of tracking..." << endl;
			/** Caching intermediate frames **/
			if (queuedResponse.size() > 0){
				CachedFrame interFrame(curFrame, clock, frameName);
				cachedFrames.push_back(interFrame);
				//cout << "cache frame: " <<  frameName << endl;
			}
			/** Done with caching **/
			cout << "frameface num:" << frameFace.faces.size() << endl;
	
			/** decide if we need to send the next frame to the server 
			** should be smarter **/
			if (clock > nextWholeFrameProcessingTime){
				if (prevProcessedFrame.cols == 0 || prevProcessedFrame.rows == 0){
					curFrame.copyTo(prevProcessedFrame);
					processedRegionTracker.init(prevProcessedFrame);
				}else{
					int estimatedTrackingTime = 0;
					curFrame.copyTo(prevProcessedFrame);
					processedRegionTracker.track(curFrame, estimatedTrackingTime);
					nextWholeFrameProcessingTime += estimatedTrackingTime;
					// check if we need to do motion analysis
					if (processedRegionTracker.isLost(trackedRegion)){ //yes, we need to do motion analysis
						
						prevProcessedFrame.release();
						trackingState = 0;
					}
					Scalar color = colors[4%8];
					rectangle(curFrame, cvPoint(trackedRegion.x, trackedRegion.y), cvPoint((trackedRegion.x + trackedRegion.width), trackedRegion.y + trackedRegion.height),
					color, 3, 8, 0);
					cout <<"trackedReegion:" << trackedRegion.x << ","<< trackedRegion.y << "," << trackedRegion.x + trackedRegion.width << "," << trackedRegion.y + trackedRegion.height << endl;
					imshow("trackedRegion", curFrame);
					waitKey(60);
				}
			}
			/*****/
			if (clock - prevSentTime > checkPeriodMs && queuedResponse.size() == 0){
				operation = 7;
				trackingState = 0;
			}	
			
	}
	/** Done with all different methods**/

		double t1 = Utility::getCurrentTimeMSV2();
		double totalProcessTime = 0.0;
		if (mode == SERVERONLY || mode == GRAYMOTIONSERVER){
			totalProcessTime =  (t1 - t0) - extraSendTime - redundantSleepingTime + devicePenaltyTime;
		}else if(mode == SIMPLETRACKING || mode == FORWARDTRACKING || mode == SLIDE ||  mode == IMAGEREGISTRATION || mode == FORWARDTRACKINGEWMA|| mode == JPEGCOMPRESS || mode == HARIBASELINE){
			totalProcessTime = (t1 - t0) + devicePenaltyTime - extraSendTime;
		}
		/*** yuhan ***/
		//double totalProcessTime = (t1 - t0) + devicePenaltyTime - extraSendTime;
		cout << "t1-t0:" << (t1-t0) << ", devicePenalty:" << devicePenaltyTime <<" ,extraSentTime:"  << extraSendTime <<endl;
		cout << "Processing time = " <<  totalProcessTime << " ms" << endl;
		/**
		* Plot the results
		*/ 
		for (int i = 0; i < frameFace.faces.size(); ++i){
			Rect r = frameFace.faces.at(i).faceRect;
			Scalar color = colors[i%8];
			rectangle(curFrame, cvPoint(r.x, r.y), cvPoint((r.x + r.width), r.y + r.height),
				 color, 3, 8, 0);
			
			for (int j = 0; j < frameFace.faces.at(i).featurePoints.size(); ++j){
				circle(curFrame, frameFace.faces.at(i).featurePoints.at(j), 2, CV_RGB(255,0,0),1);
			}
			// then put the text itself
			int fontFace = FONT_HERSHEY_SIMPLEX;
			double fontScale = 1;
			int thickness = 1;
			putText(curFrame, labelToSubj.at(frameFace.faces.at(i).label) + "," + to_string(frameFace.faces.at(i).confidence), cvPoint(r.x-10, r.y-19), fontFace, fontScale,
			Scalar::all(255), thickness, 8);
		}
		
		cv::imshow( accuracyKey, curFrame );
		cv::waitKey( 10 );
		//getchar();

		/**
		* Compute statistics
		**/
		// delay, (done with processing - spit) 
		double delay = (clock + totalProcessTime) - captureTime ;
		

		FrameFaceClass gt = gtMap[accuracyKey + "\\" + frameName];
		//cout << "key:" << accuracyKey + "\\" + frameName << endl;
		// compute intersect
		bool bitMap[225];
		for (int i = 0; i < 225; ++i){
			bitMap[i] = false;
		}
		for (int i = 0; i < frameFace.faces.size(); ++i){
			for (int j = 0; j < gt.faces.size(); ++j){
				if (frameFace.faces.at(i).label == gt.faces.at(j).label && bitMap[frameFace.faces.at(i).label] == false){
						bitMap[frameFace.faces.at(i).label] = true;
						++correct;
						// correctly labeled face //
						// compute bounding box region //
						hariIntersectArea.push_back(Utility::commonApBPortion(gt.faces.at(j).faceRect, frameFace.faces.at(i).faceRect));
						double overlapR = Utility::commonAuBPortion(gt.faces.at(j).faceRect, frameFace.faces.at(i).faceRect);
						boxIntersectArea.push_back(overlapR);
						frameFace.faces.at(i).overlapRatio = overlapR;
						double dist = Utility::myDistance(gt.faces.at(j).faceRect.x + gt.faces.at(j).faceRect.width/2, 
							gt.faces.at(j).faceRect.y + gt.faces.at(j).faceRect.height/2,
							frameFace.faces.at(i).faceRect.x + frameFace.faces.at(i).faceRect.width/2, 
							frameFace.faces.at(i).faceRect.y + frameFace.faces.at(i).faceRect.height/2);
						aveBoundingBoxDist.push_back(dist);
						break;
				}
			}
		}
		precisionTotal += frameFace.faces.size();
		recallTotal += gt.faces.size();
		totalTransmittedByte += transmittedByte;
		aveDelay.push_back(delay);

		if (gotoServer){
			gotoserver_fh << sentData << endl;
		}else{
			tracking_fh << frameName << "," << totalProcessTime << endl;
		}

		delay_fh << frameName << ";" << delay << ";" << transmittedByte << ";" << totalProcessTime << ";" << (t1 - t0) << ";"
			 << ";" << redundantSleepingTime << ";" << devicePenaltyTime << ";" << endl;
		//frameAccuracy_fh << frameName << ";" << (frame_correct / (frame_total * 1.0)) * 100 << "," << frame_correct << "," << frame_total << endl;
		string outputStr = frameName + ";" + to_string(frameFace.faces.size()) + ";";
		vector<int> printedFace;
		for (int i = 0; i < frameFace.faces.size(); ++i){
			if (i == 0){
				printedFace.push_back(frameFace.faces.at(i).label);
				
				 outputStr +=  frameFace.faces.at(i).toString();
			}else{
				int found = false;
				for (int j = 0; j < printedFace.size(); ++j){
					if(printedFace.at(j) == frameFace.faces.at(i).label){
						found = true;
					}
				}
				if (!found){		
					outputStr += ";" + frameFace.faces.at(i).toString();
				}
			} 
		}
		predictionLog_fh << outputStr << endl;
		
		
		// Total processing time < 1/frameRate * 1000, we need make sure we get a new frame
		if ( totalProcessTime <  1000.0/(frameRate * 1.0)){
			double frameInterval = 1000.0/(frameRate * 1.0);
			clock = ((((clock + totalProcessTime) / frameInterval) + 1.0) * frameInterval) + 1;
			//clock += 1000.0/(frameRate * 1.0);
		}else{
			clock += totalProcessTime;
		}
		//cout << "Final clock:" <<  clock << endl;
		
		//clock += 1000.0/(frameRate * 1.0);
	}

	// print statistics
	double delay_ave = 0.0;
	double delay_sum = 0.0;
	for (int i = 0; i < aveDelay.size(); ++i){
		delay_sum += aveDelay.at(i);
	}
	delay_ave = delay_sum / (aveDelay.size()*1.0);

	double MOTA = 0.0;
	double hariIntersectRatioSum = 0.0;
	double boxIntersectRatioSum = 0.0;
	double hariIntersectRatioAve = 0.0;
	double boxIntersectRatioAve = 0.0;
	for (int i = 0; i < aveBoundingBoxDist.size(); ++i){
		MOTA += aveBoundingBoxDist.at(i);
		hariIntersectRatioSum += hariIntersectArea.at(i);
		boxIntersectRatioSum += boxIntersectArea.at(i);
	}
	MOTA = aveBoundingBoxDist.size() == 0? 0 : MOTA / (aveBoundingBoxDist.size() * 1.0);
	hariIntersectRatioAve = hariIntersectArea.size() == 0? 0 : hariIntersectRatioSum/(hariIntersectArea.size()*1.0);
	boxIntersectRatioAve = boxIntersectArea.size() == 0? 0 : boxIntersectRatioSum / (boxIntersectArea.size() * 1.0);

	cout << "Precision: " << (correct / (precisionTotal * 1.0)) * 100 << " Recall:" << ( correct / (recallTotal * 1.0)) * 100<< endl;
	cout << "skiped rate:" << 100 - (totalProcessedFrame / (spitter.totalFrameNum * 1.0)) * 100 << endl;
	cout << "effective frame rate:" << totalProcessedFrame / (clock/(1000 * 1.0)) << endl;
	cout << "transmitted rate(byte/s):" << (totalTransmittedByte /(clock/(1000 * 1.0)))  << endl;
	cout << "aveDelay:" << delay_ave << endl;
	cout << "MOTA:" << MOTA << endl;
	cout << "hariIntersectRatio:" << hariIntersectRatioAve << endl;
	cout << "boxIntersectRatio:" << boxIntersectRatioAve << endl;

	accuracy_fh << "Average delay(ms):" << delay_ave << ";" << delay_sum << "/" << aveDelay.size() <<endl;
	accuracy_fh << "Precision:" << (correct / (precisionTotal * 1.0)) * 100 << ";" << correct  <<"/" << precisionTotal  << endl;
	accuracy_fh << "Recall:" <<  ( correct / (recallTotal * 1.0)) * 100 << ";" << correct << "/" << recallTotal<< endl;
	accuracy_fh << "transmitted rate(byte/s):" << (totalTransmittedByte /(clock/(1000 * 1.0))) << ";" << totalTransmittedByte << ";" << (clock/(1000 * 1.0)) << endl;
	accuracy_fh << "transmitted rate(Kbps):" << ((totalTransmittedByte /(clock/(1000 * 1.0))) * 8)/1000.0 << ";" << totalTransmittedByte << ";" << (clock/(1000 * 1.0)) << endl;
	accuracy_fh << "IntersectRatio (%):" << boxIntersectRatioAve << ";" << boxIntersectRatioSum << "/" << boxIntersectArea.size() << endl;
	accuracy_fh << "Localization Error(px):" << MOTA << endl;
	accuracy_fh << "APB IntersectReatio (%):" << hariIntersectRatioAve << ";" << hariIntersectRatioSum  << "/" << hariIntersectArea.size() << endl;
	accuracy_fh << "Frame rate:" << totalProcessedFrame / (clock/(1000 * 1.0))  <<";" << totalProcessedFrame << "/"  << (clock/(1000 * 1.0)) << endl;
	

	accuracy_fh.close();
	predictionLog_fh.close();
	tracking_fh.close();
	gotoserver_fh.close();
	//frameAccuracy_fh.close();
	delay_fh.close();
	nwk.sendGoodBye();
	nwk.close();

}