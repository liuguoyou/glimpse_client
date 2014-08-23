#include "Utility.h"
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <random>
#include "Knobs.h"

void Utility::convertJpegEncodedBuffToMat(char* buffer, int w, int h, cv::Mat& decodedImg){

	cv::Mat tmpFrame(h, w, CV_8UC1, buffer);
	decodedImg = cv::imdecode(tmpFrame , 0);
}
int Utility::getMotionAnalysisTime(){
	//18.38773841	10.70789749

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(18.38773841,10.70789749);
	
	double number = distribution(generator);
	return (int)number;
}
int Utility::getCompressionTime(int width, int height){
/**
1	39.18324655	11.37283646
0.9	32.8273329	8.955634126
0.8	24.40313292	4.606156214
0.7	19.31164166	5.803743807
0.6	14.68328428	5.002133174
0.5	10.44276036	4.54102539
0.4	7.099760739	4.618348956
0.3	3.986636612	3.150808856
0.2	1.843167113	1.575336199
0.1	0.807698597	2.230989416
**/
	float mean[] = {0, 0.807698597,1.843167113	, 3.986636612, 7.099760739,10.44276036,14.68328428,19.31164166,24.40313292,32.8273329,39.18324655};	
    float std[] = {0, 2.230989416,1.575336199, 3.15080885, 4.618348956, 4.54102539, 5.002133174,5.803743807,4.606156214, 8.955634126, 11.37283646};
	float area = width * height;
	int ratio = 0;
	int ratioBin = cvRound(10 * sqrt(area / float(640 * 480)));
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::normal_distribution<double> distribution(mean[ratioBin],std[ratioBin]);
	
	double number = distribution(generator);
	return (int)number;
}
double Utility::myDistance(double x1, double y1, double x2, double y2){
	return sqrt((x1 - x2) * (x1 - x2) +(y1- y2) * (y1- y2));
}
std::vector<std::string> Utility::list_files(std::string dirName){
	int count = 0;
	DIR *dir;
	struct dirent *ent;
	std::vector<std::string> files;


	/* Open directory stream */
	dir = opendir (dirName.c_str());
	if (dir != NULL) {
		
		/* Print all files and directories within the directory */
		while ((ent = readdir (dir)) != NULL) {
			if (ent->d_type == DT_REG){
				++count;
			}
		}
		rewinddir(dir);
		files.reserve(count);
		

		while ((ent = readdir (dir)) != NULL) {
			if (ent->d_type == DT_REG){
				std::string fileName = _strdup(ent->d_name);
				std::size_t found  = fileName.find("db");
				if (found==std::string::npos)
					files.push_back(fileName);
			}
		}
		closedir (dir);
		
		

	} else {
		/* Could not open directory */
		printf ("Cannot open directory %s\n", dirName);
		exit (EXIT_FAILURE);
	}
	std::sort(files.begin(), files.end(), Utility::wayToSort);
	return files;
}

bool Utility::wayToSort(std::string i, std::string j) {

	std::vector<std::string> i_segs = Utility::split(i, ".");
	std::vector<std::string> j_segs = Utility::split(j, ".");

	if (i_segs.size() == 2){ // my dataset
		if (atoi(i_segs[0].c_str()) <  atoi(j_segs[0].c_str())){
			return 1;
		}
	}
	else if(i_segs.size() == 3){ // youtube dataset
		if (atoi(i_segs[1].c_str()) <  atoi(j_segs[1].c_str())){
			return 1;
		}
	}
	return 0;

}
std::vector<std::string> Utility::split(std::string str, std::string delim){

	size_t start = 0;
	size_t end;
	std::vector<std::string> v;
	
	while( (end = str.find(delim, start)) != std::string::npos ){
		v.push_back(str.substr(start, end-start));
		start = end + delim.length();
	}
	v.push_back(str.substr(start));
	return v;
}

std::map<std::string,int> Utility::ReadSubjs(std::string filePath, std::vector<std::string>& labelToSubj){
	std::map<std::string,int> subs;
	std::ifstream in;
	in.open(filePath);
	if ( ! in ) {
		printf("Error: Can't open the file named %s.\n", filePath);
		exit(EXIT_FAILURE);
	}
	int index = 0;
	int counter = 0;
	std::string str;
	while ( in ) {  // Continue if the line was sucessfully read.
		getline(in,str); 
		labelToSubj.push_back(str);
		subs.insert(make_pair(str,index));
		++index;
	}
	in.close();
	return subs;
}

double Utility::getCurrentTimeMS() {
    LARGE_INTEGER s_frequency;
	LARGE_INTEGER now;
    QueryPerformanceFrequency(&s_frequency);
    QueryPerformanceCounter(&now);
    return (1000.0 * now.QuadPart) / s_frequency.QuadPart;
}

double Utility::getCurrentTimeMSV2(){
	return cv::getTickCount()/((double)cvGetTickFrequency()*1000.);
}
double Utility::commonApBPortion(cv::Rect a, cv::Rect b){ // A && B / A + B
	cv::Rect intersect = a & b;
	
	double area_a = a.width * a.height;
	double area_b = b.width * b.height;
	double biggerArea = area_a > area_b? area_a: area_b;
	return (intersect.width * intersect.height)/(biggerArea* (1.0));
}

int Utility::overLap(cv::Rect a, cv::Rect b){
	cv::Rect intersect = a & b;
	double area_intersect = intersect.width * intersect.height;
	double area_a = a.width * a.height;
	double area_b = b.width * b.height;

	if (area_intersect >= area_a || area_intersect >= area_b){
		return 1;
	}else{
		return 0;
	}
}
double Utility::commonAuBPortion(cv::Rect a, cv::Rect b){ // A && B / A || B
	cv::Rect intersect = a & b;
	double area_intersect = intersect.width * intersect.height;
	double area_a = a.width * a.height;
	double area_b = b.width * b.height;
	return (area_intersect)/((area_a + area_b - area_intersect) * 1.0);
}
void Utility::enlargeROI(cv::Mat curFrame, cv::Rect suggestROI, cv::Mat& croppedEnlargedROI, cv::Rect& enlargedROI){

	int mid_x = suggestROI.x + suggestROI.width/2;
	int mid_y = suggestROI.y + suggestROI.height/2;
	// check boundary
	int right_width = suggestROI.width;
	int left_width = suggestROI.width;
	int upper_height = suggestROI.height;
	int lower_height = suggestROI.height;

	if ((mid_x + right_width) > curFrame.cols){
		right_width = curFrame.cols - mid_x;
	}
	if ((mid_x - left_width) < 0){
		left_width = mid_x;
	}
	if ((mid_y + lower_height) > curFrame.rows){
		lower_height = curFrame.rows - mid_y;
	}
	if ((mid_y - upper_height) < 0){
		upper_height = mid_y;
	}
	enlargedROI.x =  mid_x - left_width;
	enlargedROI.y = mid_y - upper_height;
	enlargedROI.width = left_width + right_width;
	enlargedROI.height = upper_height + lower_height;

	Knobs::cropFrame(curFrame, croppedEnlargedROI, mid_x - left_width, mid_y - upper_height, left_width + right_width, upper_height + lower_height);
							
}