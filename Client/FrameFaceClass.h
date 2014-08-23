#ifndef FRAMEFACECLASS_H
#define FRAMEFACECLASS_H
#include <vector>
#include "FaceClass.h"

class FrameFaceClass{
public:
	int faceNum;
	std::vector<FaceClass> faces;

	
	FrameFaceClass()
	: faceNum(-1)
	, faces(NULL)
	{}

};
#endif