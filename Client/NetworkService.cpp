#include <winsock2.h>
#include <windows.h>
#include <sstream>
#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>
#include "NetworkService.h"
#include "ResponseClass.h"
#include "Utility.h"

#pragma comment(lib,"ws2_32.lib") //Winsock Library

NetworkService::NetworkService(int port, char* ipAddress){
	
    // create WSADATA object
    WSADATA wsa;
    
    // Initialize Winsock
	printf("Init Winsock...\n");
    if( WSAStartup(MAKEWORD(2,2), &wsa) != 0){
		printf("Failed. Error Code : %d",WSAGetLastError());
        exit(EXIT_FAILURE);
	}
	
    //Fill out the information needed to initialize a socket
    struct sockaddr_in server; //Socket address information

    server.sin_family = AF_INET; // address family Internet
	server.sin_addr.s_addr = inet_addr (ipAddress); //Target IP
    server.sin_port = htons (port); //Port to connect on
    
	if((connectSocket = socket (AF_INET, SOCK_STREAM, IPPROTO_TCP)) == INVALID_SOCKET)
    {
        printf("Could not create socket : %d" , WSAGetLastError());
		exit(EXIT_FAILURE);
    }     
    
	//-------------------------
	// Set the socket I/O mode: In this case FIONBIO
	// enables or disables the blocking mode for the 
	// socket based on the numerical value of iMode.
	// If iMode = 0, blocking is enabled; 
	// If iMode != 0, non-blocking mode is enabled.
	//-------------------------
	u_long iMode = 0;
	if (ioctlsocket(connectSocket, FIONBIO, &iMode) != NO_ERROR){
		printf("ioctlsocket failed with error: %d\n",WSAGetLastError());
 		exit(EXIT_FAILURE);
	}

	 //Connect to remote server
    if (connect(connectSocket , (struct sockaddr *)&server , sizeof(server)) < 0)
    {
        puts("connect error");
        exit(EXIT_FAILURE);
    }

	int recv_size;
	char connect_msg[15];
	// Receive welcome msg
	if((recv_size = recv(connectSocket , connect_msg , sizeof(connect_msg) , 0)) == SOCKET_ERROR)
	{
		puts("recv header failed");
		exit(EXIT_FAILURE);
	}
	connect_msg[recv_size] = '\0';
	printf("%s\n", connect_msg);
}

void NetworkService::sendProcessFrameHeader(int faceID, int imgSize, int imgWidth, int imgHeight, double delta){
	
	std::stringstream strs;
	std::stringstream ss;
	ss << std::fixed << delta;
	std::stringstream ss_2;
	ss_2 << std::fixed << Utility::getCurrentTimeMSV2();

	strs << "0;" << faceID << ";" << imgSize << ";" << imgWidth << ";" << imgHeight << ";" << ss.str() << ";" << ss_2.str() << std::endl;
	std::string tmp_str = strs.str();
	char* hearder_msg = (char *)tmp_str.c_str();
	printf("%s\n",hearder_msg);
	if ( send(connectSocket, hearder_msg, strlen(hearder_msg), 0 ) < 0){
		printf("sendProcessFrameHeader failed\n");
		exit(EXIT_FAILURE);
	}
	

}
void NetworkService::sendProcessFaceHeader(const FaceClass fc, int imgSize, int imgWidth, int imgHeight, double delta){
	
	std::stringstream strs;
	strs << "1;" << fc.faceID << ";" << imgSize << ";" << imgWidth << ";" << imgHeight << ";" << fc.faceRect.x << ";" << fc.faceRect.y << std::endl;
	std::string tmp_str = strs.str();
	char* hearder_msg = (char *)tmp_str.c_str();
	printf("%s\n",hearder_msg);
	if ( send(connectSocket, hearder_msg, strlen(hearder_msg), 0 ) < 0){
		printf("sendProcessFrameHeader failed\n");
		exit(EXIT_FAILURE);
	}
	
}
void NetworkService::sendEntireFrame(char* frame, int size, std::vector<EntireFrameResponse> responses){
	
	// Send frame
	double t = Utility::getCurrentTimeMSV2();
	if ( send(connectSocket, frame, size, 0) < 0){
		printf("sendFrame failed\n");
		exit(EXIT_FAILURE);
	}

	// Receive response msg
	int recv_size;
	char response_msg[2046];	
	if((recv_size = recv(connectSocket , response_msg , sizeof(response_msg) , 0)) == SOCKET_ERROR)
	{
		puts("recv sendFrame response failed");
		exit(EXIT_FAILURE);
	}
	double totalProcessingTime = Utility::getCurrentTimeMSV2() - t;
	
	response_msg[recv_size] = '\0';
	//printf("response: %s\n", response_msg);

	std::string response_str = response_msg;
	EntireFrameResponse response(response_str, totalProcessingTime);


}
EntireFrameResponse NetworkService::sendEntireFrame(char* frame, int size){

	// Send frame
	double t = Utility::getCurrentTimeMSV2();
	if ( send(connectSocket, frame, size, 0) < 0){
		printf("sendFrame failed\n");
		exit(EXIT_FAILURE);
	}	

	// Receive response msg
	int recv_size;
	char response_msg[2046];	
	if((recv_size = recv(connectSocket , response_msg , sizeof(response_msg) , 0)) == SOCKET_ERROR)
	{
		puts("recv sendFrame response failed");
		exit(EXIT_FAILURE);
	}
	double totalProcessingTime = Utility::getCurrentTimeMSV2() - t;
	
	response_msg[recv_size] = '\0';
	//printf("response: %s\n", response_msg);

	std::string response_str = response_msg;
	EntireFrameResponse response(response_str, totalProcessingTime);

	return response;
}

EntireFrameResponse NetworkService::sendFace(char* frame, int size){

	// Send frame
	double t = Utility::getCurrentTimeMSV2();
	if ( send(connectSocket, frame, size, 0) < 0){
		printf("sendFrame failed\n");
		exit(EXIT_FAILURE);
	}	

	// Receive response msg
	int recv_size;
	char response_msg[2046];	
	if((recv_size = recv(connectSocket , response_msg , sizeof(response_msg) , 0)) == SOCKET_ERROR)
	{
		puts("recv sendFrame response failed");
		exit(EXIT_FAILURE);
	}
	double totalProcessingTime = Utility::getCurrentTimeMSV2() - t;
	response_msg[recv_size] = '\0';
	//printf("%s\n", response_msg);

	std::string response_str = response_msg;
	EntireFrameResponse response(response_str, totalProcessingTime);

	return response;
}

EntireFrameResponse NetworkService::sendEntireFrame(cv::Mat frame){

	// Send frame
	double t = Utility::getCurrentTimeMSV2();
	int img_size = frame.total() * frame.elemSize();	
	if ( send(connectSocket, (char *)frame.data, img_size, 0) < 0){
		printf("sendFrame failed\n");
		exit(EXIT_FAILURE);
	}	

	// Receive response msg
	int recv_size;
	char response_msg[2046];	
	if((recv_size = recv(connectSocket , response_msg , sizeof(response_msg) , 0)) == SOCKET_ERROR)
	{
		puts("recv sendFrame response failed");
		exit(EXIT_FAILURE);
	}
	double totalProcessingTime = Utility::getCurrentTimeMSV2() - t;
	response_msg[recv_size] = '\0';
	//printf("%s\n", response_msg);

	std::string response_str = response_msg;
	EntireFrameResponse response(response_str, totalProcessingTime);

	return response;
}
void NetworkService::sendGoodBye(){

	char* goodBye = "-1";
	if ( send(connectSocket, goodBye, strlen(goodBye), 0 ) < 0){
		printf("sendProcessFrameHeader failed\n");
		exit(EXIT_FAILURE);
	}
}
double NetworkService::sendSync(){

	char* syncCommand = "2";
	if ( send(connectSocket, syncCommand, strlen(syncCommand), 0 ) < 0){
		printf("sendSync failed\n");
		exit(EXIT_FAILURE);
	}
	int recv_size;
	char sync_msg[100];
	// Receive welcome msg
	if((recv_size = recv(connectSocket , sync_msg , sizeof(sync_msg) , 0)) == SOCKET_ERROR)
	{
		puts("recv timesync failed");
		exit(EXIT_FAILURE);
	}
	sync_msg[recv_size] = '\0';
	printf("%s\n", sync_msg);
	return atof(sync_msg);
}
void NetworkService::close(){
	closesocket(connectSocket);
    WSACleanup();
}