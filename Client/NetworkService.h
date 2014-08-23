#include <winsock2.h>
#include <windows.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "ResponseClass.h"
#pragma comment(lib,"ws2_32.lib") //Winsock Library

class NetworkService{

public:
	
    // socket for client to connect to server
    SOCKET connectSocket;

    NetworkService(int port, char* ipAddress);	
	void sendProcessFrameHeader(int faceID, int imgSize, int imgWidth, int imgHeight, double delta);
	void sendProcessFaceHeader(const FaceClass fc, int imgSize, int imgWidth, int imgHeight, double delta);
	EntireFrameResponse sendEntireFrame(char* frame, int size);
	void NetworkService::sendEntireFrame(char* frame, int size, std::vector<EntireFrameResponse> responses);
	EntireFrameResponse sendFace(char* frame, int size);
	EntireFrameResponse sendEntireFrame(cv::Mat);
	void sendGoodBye();
	double sendSync();
	void close();
    //~NetworkService(int port, char* ipAddress);
	

};