//
//  FaceDetect.h
//  NodeModuleTest
//
//  Created by 叶 晗 on 13-2-23.
//  Copyright (c) 2013年 ZJU. All rights reserved.
//

#ifndef NodeModuleTest_FaceDetect_h
#define NodeModuleTest_FaceDetect_h

extern "C" int facedetect(char* inputName);
extern "C" int buildfacedata();
extern "C" int facerec(char* inputName);
extern "C" char* hello();
#endif
