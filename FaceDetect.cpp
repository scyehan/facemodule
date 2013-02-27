//
//  FaceDetect.cpp
//  NodeModuleTest
//
//  Created by 叶 晗 on 13-2-23.
//  Copyright (c) 2013年 ZJU. All rights reserved.
//

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv/cv.h"

#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/types.h>
#include "hiredis/hiredis.h"
#include "cJSON.h"

#include "FaceDetect.h"

using namespace std;
using namespace cv;

#define FIX_PIC_WIDTH 1280.0
#define USER_SET_PATH "test:userset"
#define NAME_MAP_PATH "test:namemap"
#define HEADER_PREFIX "test:headers:"

void GetImageRect(IplImage* orgImage, CvRect rectInImage, IplImage* imgRect);

const string cascadeName = "haarcascade_frontalface_alt2.xml";

int facedetect(char* fileName)
{
    string inputPath = "public/uploads/";
    inputPath += fileName;
	Mat img = imread( inputPath.c_str(), 1 );
    if(img.empty())
        return -1;
    IplImage * jpgImg = cvLoadImage(inputPath.c_str(),CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
	CascadeClassifier cascade;
	cascade.load( cascadeName );
    int i = 0;
    double scale = img.rows / FIX_PIC_WIDTH;
    if (scale < 1) {
        scale = 1;
    }
    vector<Rect> faces;
	Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    
    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    cascade.detectMultiScale( smallImg, faces,
                             1.1, 2, 0
                             //|CV_HAAR_FIND_BIGGEST_OBJECT
                             //|CV_HAAR_DO_ROUGH_SEARCH
                             |CV_HAAR_SCALE_IMAGE
                             ,
                             Size(30, 30) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
		CvSize size;
		size.width = cvRound(r->width * scale);
		size.height = cvRound(r->height * scale);
		IplImage* result=cvCreateImage( size, jpgImg->depth, jpgImg->nChannels );
        size.width = cvRound(200);
		size.height = cvRound(200);
        IplImage* result2=cvCreateImage( size, jpgImg->depth, jpgImg->nChannels );
        CvRect  cvRect;
        cvRect.x = cvRound(r->x * scale);
        cvRect.y = cvRound(r->y * scale);
        cvRect.width = cvRound(r->width * scale);
        cvRect.height = cvRound(r->height * scale);
		GetImageRect(jpgImg, cvRect, result);
		cvResize( result, result2);
		char outPutPath[50];
		sprintf(outPutPath,"public/headers/%s_%d.jpg",fileName,i);
		cvSaveImage(outPutPath,result2);
    }
    
	return i;
}

static void read_dir(const char* dirname, vector<Mat>& images, vector<int>& labels,int value)
{
    DIR* dir;
    struct dirent* ptr;
    string path = "rec/";
    path += dirname;
    dir = opendir(path.c_str());
    
    while ((ptr = readdir(dir))!= NULL) {
        if(strcmp(ptr->d_name, ".") == 0)
            continue;
        if(strcmp(ptr->d_name, "..") == 0)
            continue;
        if(strcmp(ptr->d_name, ".DS_Store") == 0)
            continue;
        cout << dirname << "  " << ptr->d_name << endl;
        char tmp[50];
        sprintf(tmp,"%s/%s",path.c_str(),ptr->d_name);
        images.push_back(imread(tmp,CV_LOAD_IMAGE_GRAYSCALE));
        labels.push_back(value);
    }
}

int buildfacedata()
{
    vector<Mat> images;
    vector<int> labels;
//    read_dir("1",images,labels,1);
//    read_dir("2",images,labels,2);
//    read_dir("3",images,labels,3);
//    read_dir("4",images,labels,4);
    
    redisContext *redis = redisConnect("localhost", 6379);
    redisReply *userSetReply = (redisReply*)redisCommand(redis, "smembers %s",USER_SET_PATH);
    
    if (userSetReply->elements > 0) {
        cJSON* object = cJSON_CreateObject();
        for (int i=0; i<userSetReply->elements; i++) {
            redisReply *headerSetReply = (redisReply*)redisCommand(redis, "smembers %s%s",HEADER_PREFIX,userSetReply->element[i]->str);
            for (int j=0; j<headerSetReply->elements; j++) {
                //string path = "public/headers/";
                //path += headerSetReply->element[j]->str;
                Mat mat = imread(headerSetReply->element[j]->str,CV_LOAD_IMAGE_GRAYSCALE);
                if(mat.empty())
                    continue;
                images.push_back(mat);
                labels.push_back(i);
                if(j == 0)
                {
                    char* name = userSetReply->element[i]->str;
                    char num[3];
                    sprintf(num, "%d",i);
                    cJSON_AddItemToObject(object, num, cJSON_CreateString(name));
                }
            }
        }
        
        char* nameMapStr = cJSON_PrintUnformatted(object);
        cout << nameMapStr << endl;
        redisCommand(redis, "set %s %s",NAME_MAP_PATH,nameMapStr);
        Ptr<FaceRecognizer> model0 = createFisherFaceRecognizer();
        model0->train(images, labels);
        model0->save("facedata.yml");
        cout << "build data success!" << endl;
        return 1;
    }
    return 0;
}

int facerec(char* inputName)
{
    Ptr<FaceRecognizer> model1 = createFisherFaceRecognizer(0,800);
    model1->load("facedata.yml");
    string path = "public/headers/";
    path += inputName;
    Mat testSample = imread( path, CV_LOAD_IMAGE_GRAYSCALE );
    int predictedLabel = model1->predict(testSample);
    return predictedLabel;
}

void GetImageRect(IplImage* orgImage, CvRect rectInImage, IplImage* imgRect)
{
    
    IplImage *result=imgRect;
    CvSize size;
    size.width=rectInImage.width;
    size.height=rectInImage.height;
    //result=cvCreateImage( size, orgImage->depth, orgImage->nChannels );
    
    cvSetImageROI(orgImage,rectInImage);
    cvCopy(orgImage,result);
    cvResetImageROI(orgImage);
}