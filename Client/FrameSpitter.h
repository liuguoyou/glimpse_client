#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

class FrameSpitter{
private:
	int frameRate;
	std::string videoFrameFolder;
	std::vector<std::string> frames;
	

public:
	int totalFrameNum;
	FrameSpitter(std::string videoFrameFolder, int _frameRate);
	void loadFrames();
	bool has_frame(double time);
	cv::Mat spit(double time, int &frameIndex, double &captureTime, std::string& frameName);
};