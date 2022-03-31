#include <iostream>
#include <fstream>
#include <stdio.h>
#include<math.h>
#include<stdlib.h>
#include <time.h>
#include <vector>
#include <iomanip>
#include <opencv2\opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>

using namespace cv;
using namespace std;



class ChangSul {
private:
	int *locations_x;
	int *locations_y;

public:
	void BlueTone(const char*file) {
		CvCapture *capture = cvCaptureFromFile(file);
		if (!capture) {
			std::cout << "The video file was not found.";
		}
		int width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		CvSize frameSize = cvSize(width, height);

		IplImage *redImage = cvCreateImage(frameSize, IPL_DEPTH_8U, 1);
		IplImage *greenImage = cvCreateImage(frameSize, IPL_DEPTH_8U, 1);
		IplImage *blueImage = cvCreateImage(frameSize, IPL_DEPTH_8U, 1);
		IplImage *sumImage = cvCreateImage(frameSize, IPL_DEPTH_32F, 1);

		int ***arr;
		arr = new int **[256];
		int q, g, p;
		for (int q = 0; q < 256; q++){
			arr[q] = new int*[width];
			for (int g = 0; g < width; ++g){
				arr[q][g] = new int[height];
			}
		}
		for (q = 0; q < 256; q++){
			for (g = 0; g < width; ++g){
				for (p = 0; p < height; ++p){
					arr[q][g][p] = 0;
				}
			}
		}

		cvZero(sumImage);

		int x, y;
		int value;

		IplImage *frame = NULL;
		int t = 0;
		while (1) {
			// capture로부터 프레임을 획득하여 포인터 frame에 저장한다.
			frame = cvQueryFrame(capture);
			if (!frame)    break;
			t++;
			std::cout << t << std::endl;

			//cVsplit으로 RGB 채널 분리
			cvSplit(frame, blueImage, greenImage, redImage, 0);

			/*// cvAcc 함수를 사용하여 blueImage 영상을 sumImage에 누적한다
			cvAcc(blueImage, sumImage, NULL);*/
			cvShowImage("blueImage", blueImage);
			for (x = 0; x < width; ++x){
				for (y = 0; y < height; ++y){
					value = cvGetReal2D(blueImage, y, x);
					arr[value][x][y] += 1;
				}
			}
			char chKey = cvWaitKey(10);
			if (chKey == 27)    // ESC    
				break;
		}
		// cvScale 함수를 사용하여 누적 영상 sumImage를 1.0/t로 스케일링하여 평균 영상을 계산하여 sumImage에 다시 저장
		//cvScale(sumImage, sumImage, 1.0 / t);
		Mat sum;
		int max, cnt;
		for (x = 0; x < width; ++x){
			for (y = 0; y < height; ++y){
				max = 0;
				cnt = 0;
				for (q = 0; q < 256; ++q){
					if (arr[q][x][y]>cnt){
						cnt = arr[q][x][y];
						max = q;
						}					
				}
				std::cout << x << "  " << y << "  " << max << std::endl;
				cvSet2D(sumImage, y, x, cvScalar(max));
			}
		}
		for (q = 0; q < 256; ++q)
		{
			for (x = 0; x < width; x++)
			{
				delete[] arr[q][y];
			}
		}

		for (q = 0; q < 256; ++q)
		{
			delete[] arr[q];
		}
		delete[] arr;
		// cvSaveImage 함수를 이용하여 sumImage에 저장된 평균 영상을 저장한다.
		//    cvSaveImage("ballBkg.jpg", sumImage);
		//    cvSaveImage("carBkg.jpg", sumImage);
		//    cvSaveImage("tunnelBkg.jpg", sumImage);
		// cvCvtColor(sumImage, testImage, CV_GRAY2BGR);
		cvSaveImage("exBkg_blue.jpg", sumImage);

		cvDestroyAllWindows();
		cvReleaseImage(&sumImage);
		cvReleaseImage(&redImage);
		cvReleaseImage(&greenImage);
		cvReleaseImage(&blueImage);
		cvReleaseCapture(&capture);

	}
	void GreenTone(const char*file) {
		CvCapture *capture = cvCaptureFromFile(file);
		if (!capture) {
			std::cout << "The video file was not found.";
		}
		int width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		CvSize frameSize = cvSize(width, height);

		IplImage *redImage = cvCreateImage(frameSize, IPL_DEPTH_8U, 1);
		IplImage *greenImage = cvCreateImage(frameSize, IPL_DEPTH_8U, 1);
		IplImage *blueImage = cvCreateImage(frameSize, IPL_DEPTH_8U, 1);
		IplImage *sumImage = cvCreateImage(frameSize, IPL_DEPTH_32F, 1);
		int ***arr;
		arr = new int **[256];
		int q, g, p;
		for (int q = 0; q < 256; q++){
			arr[q] = new int*[width];
			for (int g = 0; g < width; ++g){
				arr[q][g] = new int[height];
			}
		}
		for (q = 0; q < 256; q++){
			for (g = 0; g < width; ++g){
				for (p = 0; p < height; ++p){
					arr[q][g][p] = 0;
				}
			}
		}

		cvZero(sumImage);

		int x, y;
		int value;

		IplImage *frame = NULL;
		int t = 0;
		while (1) {
			// capture로부터 프레임을 획득하여 포인터 frame에 저장한다.
			frame = cvQueryFrame(capture);
			if (!frame)    break;
			t++;
			std::cout << t << std::endl;

			//cVsplit으로 RGB 채널 분리
			cvSplit(frame, blueImage, greenImage, redImage, 0);

			/*// cvAcc 함수를 사용하여 blueImage 영상을 sumImage에 누적한다
			cvAcc(blueImage, sumImage, NULL);*/
			cvShowImage("greenImage", greenImage);
			for (x = 0; x < width; ++x){
				for (y = 0; y < height; ++y){
					value = cvGetReal2D(greenImage, y, x);
					arr[value][x][y] += 1;
				}
			}
			char chKey = cvWaitKey(10);
			if (chKey == 27)    // ESC    
				break;
		}
		// cvScale 함수를 사용하여 누적 영상 sumImage를 1.0/t로 스케일링하여 평균 영상을 계산하여 sumImage에 다시 저장
		//cvScale(sumImage, sumImage, 1.0 / t);
		Mat sum;
		int max, cnt;
		for (x = 0; x < width; ++x){
			for (y = 0; y < height; ++y){
				max = 0;
				cnt = 0;
				for (q = 0; q < 256; ++q){
					if (arr[q][x][y]>cnt){
						cnt = arr[q][x][y];
						max = q;
					}
				}
				std::cout << x << "  " << y << "  " << max << std::endl;
				cvSet2D(sumImage, y, x, cvScalar(max));
			}
		}


		for (q = 0; q < 256; ++q)
		{
			for (x = 0; x < width; x++)
			{
				delete[] arr[q][y];
			}
		}
		
		for (q = 0; q < 256; ++q)
		{
			delete[] arr[q];
		}
		delete[] arr;
		// cvSaveImage 함수를 이용하여 sumImage에 저장된 평균 영상을 저장한다.
		//    cvSaveImage("ballBkg.jpg", sumImage);
		//    cvSaveImage("carBkg.jpg", sumImage);
		//    cvSaveImage("tunnelBkg.jpg", sumImage);
		// cvCvtColor(sumImage, testImage, CV_GRAY2BGR);
		cvSaveImage("exBkg_green.jpg", sumImage);

		cvDestroyAllWindows();
		cvReleaseImage(&sumImage);
		cvReleaseImage(&redImage);
		cvReleaseImage(&greenImage);
		cvReleaseImage(&blueImage);
		cvReleaseCapture(&capture);
	}
	void RedTone(const char*file) {
		CvCapture *capture = cvCaptureFromFile(file);
		if (!capture) {
			std::cout << "The video file was not found.";
		}
		int width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		CvSize frameSize = cvSize(width, height);

		IplImage *redImage = cvCreateImage(frameSize, IPL_DEPTH_8U, 1);
		IplImage *greenImage = cvCreateImage(frameSize, IPL_DEPTH_8U, 1);
		IplImage *blueImage = cvCreateImage(frameSize, IPL_DEPTH_8U, 1);
		IplImage *sumImage = cvCreateImage(frameSize, IPL_DEPTH_32F, 1);
		int ***arr;
		arr = new int **[256];
		int q, g, p;
		for (int q = 0; q < 256; q++){
			arr[q] = new int*[width];
			for (int g = 0; g < width; ++g){
				arr[q][g] = new int[height];
			}
		}
		for (q = 0; q < 256; q++){
			for (g = 0; g < width; ++g){
				for (p = 0; p < height; ++p){
					arr[q][g][p] = 0;
				}
			}
		}

		cvZero(sumImage);

		int x, y;
		int value;

		IplImage *frame = NULL;
		int t = 0;
		while (1) {
			// capture로부터 프레임을 획득하여 포인터 frame에 저장한다.
			frame = cvQueryFrame(capture);
			if (!frame)    break;
			t++;
			std::cout << t << std::endl;

			//cVsplit으로 RGB 채널 분리
			cvSplit(frame, blueImage, greenImage, redImage, 0);

			/*// cvAcc 함수를 사용하여 blueImage 영상을 sumImage에 누적한다
			cvAcc(blueImage, sumImage, NULL);*/
			cvShowImage("redImage", redImage);
			for (x = 0; x < width; ++x){
				for (y = 0; y < height; ++y){
					value = cvGetReal2D(redImage, y, x);
					arr[value][x][y] += 1;
				}
			}
			char chKey = cvWaitKey(10);
			if (chKey == 27)    // ESC    
				break;
		}
		// cvScale 함수를 사용하여 누적 영상 sumImage를 1.0/t로 스케일링하여 평균 영상을 계산하여 sumImage에 다시 저장
		//cvScale(sumImage, sumImage, 1.0 / t);
		Mat sum;
		int max, cnt;
		for (x = 0; x < width; ++x){
			for (y = 0; y < height; ++y){
				max = 0;
				cnt = 0;
				for (q = 0; q < 256; ++q){
					if (arr[q][x][y]>cnt){
						cnt = arr[q][x][y];
						max = q;
					}
				}
				std::cout << x << "  " << y << "  " << max << std::endl;
				cvSet2D(sumImage, y, x, cvScalar(max));
			}
		}
		for (q = 0; q < 256; ++q)
		{
			for (x = 0; x < width; x++)
			{
				delete[] arr[q][y];
			}
		}

		for (q = 0; q < 256; ++q)
		{
			delete[] arr[q];
		}
		delete[] arr;

		// cvSaveImage 함수를 이용하여 sumImage에 저장된 평균 영상을 저장한다.
		//    cvSaveImage("ballBkg.jpg", sumImage);
		//    cvSaveImage("carBkg.jpg", sumImage);
		//    cvSaveImage("tunnelBkg.jpg", sumImage);
		// cvCvtColor(sumImage, testImage, CV_GRAY2BGR);
		cvSaveImage("exBkg_red.jpg", sumImage);

		cvDestroyAllWindows();
		cvReleaseImage(&sumImage);
		cvReleaseImage(&redImage);
		cvReleaseImage(&greenImage);
		cvReleaseImage(&blueImage);
		cvReleaseCapture(&capture);
	}
	void ImageSum() {
		IplImage *img_1, *img_2, *img_3, *img_b, *img_g, *img_r, *img_rgb;

		img_1 = cvLoadImage("exBkg_blue.jpg");
		img_2 = cvLoadImage("exBkg_green.jpg");
		img_3 = cvLoadImage("exBkg_red.jpg");
		CvSize size = cvGetSize(img_1);

		int img_w = size.width; //원본 img width size
		int img_h = size.height;//원본 img height size 3등분하기

		img_b = cvCreateImage(cvSize(img_w, img_h), 8, 3); //8비트 블루 색상 이미지 활당
		img_g = cvCreateImage(cvSize(img_w, img_h), 8, 3); //8비트 그린 색상 이미지 활당
		img_r = cvCreateImage(cvSize(img_w, img_h), 8, 3); //8비트 레드 색상 이미지 활당
		img_rgb = cvCreateImage(cvSize(img_w, img_h), 8, 3); //8비트 RGB 합성 이미지 할당
		CvScalar r, g, b;


		for (int y = 0; y<img_h; y++)           // 1/3 영역 img_b 생성한곳에 블루색 변경
		{
			for (int x = 0; x<img_w; x++)
			{
				CvScalar s = cvGet2D(img_1, y, x);
				b.val[0] = s.val[0];
				cvSet2D(img_b, y, x, b);
			}
		}
		for (int y = 0; y < img_h; y++) // 2/3 영역 img_g 생성한곳에 그린색 변경
		{
			for (int x = 0; x<img_w; x++)
			{
				CvScalar s = cvGet2D(img_2, y, x);
				g.val[1] = s.val[1];
				cvSet2D(img_g, y, x, g);
			}
		}
		for (int y = 0; y<img_h; y++) // 3/3 영역 img_r 생성한곳에 레드색 변경
		{
			for (int x = 0; x<img_w; x++)
			{
				CvScalar s = cvGet2D(img_3, y, x);
				r.val[2] = s.val[2];
				cvSet2D(img_r, y, x, r);
			}
		}

		for (int y = 0; y<img_h; y++)           // RGB 이미지 합치기
		{
			for (int x = 0; x<img_w; x++)
			{
				CvScalar blue = cvGet2D(img_b, y, x);
				CvScalar green = cvGet2D(img_g, y, x);
				CvScalar red = cvGet2D(img_r, y, x);

				CvScalar rgb;
				rgb.val[0] = blue.val[0];
				rgb.val[1] = green.val[1];
				rgb.val[2] = red.val[2];

				cvSet2D(img_rgb, y, x, rgb);
			}
		}

		cvNamedWindow("hello");
		cvShowImage("hello", img_1);
		cvShowImage("hi_rgb", img_rgb);
		cvSaveImage("ColoredBK_Sample4.jpg", img_rgb);

		// cvShowImage("hi_b",img_b);
		// cvShowImage("hi_g",img_g);
		// cvShowImage("hi_r",img_r);


		cvWaitKey();
		cvReleaseImage(&img_1);
		cvReleaseImage(&img_rgb);
		cvReleaseImage(&img_b);
		cvReleaseImage(&img_g);
		cvReleaseImage(&img_r);
	}
	void Tone(){
		const char * name = "StopCam4.mp4";
		//BlueTone(name);
		GreenTone(name);
		RedTone(name);
		ImageSum();
		StopCam(name);
	}
	void StopCam(const char*file){
		CvCapture *capture = cvCaptureFromFile(file);

		if (!capture) {
			std::cout << "The video file was not found.";
		}
		VideoCapture vc(file);
		double fps = vc.get(CV_CAP_PROP_FPS);
		int fps_int = floor(fps + 0.5);
		int delay = 1000 / fps;
		int length = vc.get(CV_CAP_PROP_FRAME_COUNT);
		std::cout << fps << std::endl;


		int width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		CvSize frameSize = cvSize(width, height);

		IplImage*image1;
		IplImage*back;
		IplImage*image_sub = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage*image_tt = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage*image_momo = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);

		IplImage*image_mask = cvCreateImage(frameSize, 8, 1);
		//IplImage*image_invmask = cvCreateImage(frameSize, 8, 1);

		image1 = cvLoadImage("ColoredBK_Sample4.jpg");
		back = cvLoadImage("out_600.png");
		std::cout << "Got Colored" << std::endl;

		Mat img = (Mat)image1;
		int x, y;
		int tr = 20;
		for (x = 0; x < img.cols; ++x){
			for (y = 0; y < img.rows; ++y){
				if (img.at<Vec3b>(y, x)[0] < tr && img.at<Vec3b>(y, x)[1] < tr && img.at<Vec3b>(y, x)[2] < tr){
					img.at<Vec3b>(y, x)[2] = 255;
				}
			}
		}

		image1 = &IplImage(img);

		IplImage *frame = NULL;
		IplImage *fake = NULL;
		Show("dd", image1);
		IplImage*image_true = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage*image_false = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		Mat pic;
		int t = 0;
		while (1) {
			clock_t startTime = clock();
			frame = cvQueryFrame(capture);
			std::cout << "Got Query" << std::endl;
			if (!frame)    break;
			std::cout << t << std::endl;

			pic = (Mat)cvCloneImage(frame);
			for (x = 0; x < img.cols; ++x){
				for (y = 0; y < img.rows; ++y){
					if (pic.at<Vec3b>(y, x)[0] < tr && pic.at<Vec3b>(y, x)[1] < tr && pic.at<Vec3b>(y, x)[2] < tr){
						pic.at<Vec3b>(y, x)[2] = 255;
					}
				}
			}

			fake = &IplImage(pic);

			//함수를 사용하여 배경제거

			cvAbsDiff(fake, image1, image_sub);

			//cvNot(image_sub, image_sub);
			cvCvtColor(image_sub, image_mask, CV_BGR2GRAY);
			cvThreshold(image_mask, image_mask,13, 255, 0);
			cvAnd(frame, frame, image_true, image_mask);
			cvMerge(image_mask, image_mask, image_mask,0, image_tt);

			cvNot(image_mask, image_mask);
			cvAnd(back, back, image_false, image_mask);
			cvMerge(image_mask, image_mask, image_mask, 0, image_momo);
			cvMin(image_true, image_tt, image_true);
			cvMin(image_false, image_momo, image_false);	
			
			/*pic = (Mat)image_true;
			for (x = 0; x < img.cols; ++x){
				for (y = 0; y < img.rows; ++y){
					if (pic.at<Vec3b>(y, x)[2] == 255){
						pic.at<Vec3b>(y, x)[2] = tr;
					}
				}
			}*/

			cvAdd(image_true, image_false, image_true);

			
			cvShowImage("Minus",image_true);

			/*string name;
			name = "Pic";
			name += std::to_string(t);
			name += ".jpg";
			imwrite(name, (Mat)image_true);*/
			t++;
			
			while (clock() - startTime < delay) {
				waitKey(1);
			}
			if (waitKey(10) == 27) break;
		}


		cvDestroyAllWindows();
		cvReleaseImage(&image1);
		cvReleaseImage(&image_sub);
		cvReleaseCapture(&capture);



	}
	void Black2Red(){
		VideoCapture vc("Sample2.mp4");
		double fps = vc.get(CV_CAP_PROP_FPS);
		int fps_int = floor(fps + 0.5);
		int delay = 1000 / fps;
		int length = vc.get(CV_CAP_PROP_FRAME_COUNT);
		std::cout << fps << std::endl;

		Mat img = imread("ColoredBK2.jpg");
		Mat pic;
		int x, y;
		int tr = 50;
		for (x = 0; x < img.cols; ++x){
			for (y = 0; y < img.rows; ++y){
				if (img.at<Vec3b>(y, x)[0] < tr && img.at<Vec3b>(y, x)[1] < tr && img.at<Vec3b>(y, x)[2] < tr){
					img.at<Vec3b>(y, x)[2] = 255;
				}
			}
		}
		imshow("AA", img);
		while (1){
			clock_t startTime = clock();
			vc >> pic;
			if (pic.empty()) break;
			for (x = 0; x < img.cols; ++x){
				for (y = 0; y < img.rows; ++y){
					if (pic.at<Vec3b>(y, x)[0] < tr && pic.at<Vec3b>(y, x)[1] < tr && pic.at<Vec3b>(y, x)[2] < tr){
						pic.at<Vec3b>(y, x)[2] = 255;
					}
				}
			}
			imshow("cc", pic);
			while (clock() - startTime < delay) {
				waitKey(1);
			}
			if (waitKey(10) == 27) break;
		}

	}

	void Depth(){
		int t= 1;
		IplImage*sample = cvLoadImage(" (1).png");
		IplImage*bef;
		IplImage*aft;
		IplImage*aft_gray = cvCreateImage(cvGetSize(sample), 8, 1);
		IplImage*img = cvCreateImage(cvGetSize(sample), IPL_DEPTH_8U, 3);
		IplImage*image_tt = cvCreateImage(cvGetSize(sample), IPL_DEPTH_8U, 3);
		std::string name1, name2;
		std::string name_wr;
		const char*name_1;
		const char*name_2;
		double pp = 0;

		while (1){
			name1 = " (";
			name1 += std::to_string(t);
			name1 += ").jpg";
			name2 = " (";
			name2 += std::to_string(t);
			name2 += ").png";
			
			name_1 = name1.c_str();
			name_2 = name2.c_str();

			bef = cvLoadImage(name_1);
			if (!bef) break;
			aft = cvLoadImage(name_2);
			cvCvtColor(aft, aft_gray, CV_BGR2GRAY);
			cvNot(aft_gray, aft_gray);
			CvScalar troll;
			troll = cvAvg(aft_gray);
			pp = troll.val[0];
			cvThreshold(aft_gray, aft_gray, pp*1.55, 255, 0);
			Show("ff", aft_gray);
			cvAnd(bef, bef, img, aft_gray);
			cvMerge(aft_gray, aft_gray, aft_gray, 0, image_tt);
			cvMin(img, image_tt, img);
			Show("AD", img);
			name_wr = "Daegari";
			name_wr += std::to_string(t);
			name_wr += ".jpg";

			imwrite(name_wr, (Mat)img);
			
			t++;

			char chKey = cvWaitKey(10);
			if (chKey == 27)    // ESC    
				break;
		}
		cvDestroyAllWindows();
		cvReleaseImage(&bef);
		cvReleaseImage(&aft);
	}
	void DepthAST(){
		//Input pictures
		Mat input=imread("Daegari1.jpg");;
		IplImage *getS = cvLoadImage("Daegari1.jpg");
		int i = 1;
		int t = 1;
		string name;
		int w = cvGetSize(getS).width;
		int h = cvGetSize(getS).height;

		Mat gray;
		Mat graybg;
		//Input Background
		ifstream inFile("output_D2.txt");
		Mat bg = imread("temp2.jpg");
		IplImage *temp1 = NULL;
		IplImage *trash1 = cvCreateImage(cvGetSize(getS), IPL_DEPTH_8U, 1);
		IplImage *temp2 = NULL;
		IplImage *trash2 = cvCreateImage(cvGetSize(getS), IPL_DEPTH_8U, 1);
		int x, y;
		IplImage *result = cvCreateImage(cvGetSize(getS), IPL_DEPTH_8U, 3); 
		IplImage *merge = cvCreateImage(cvGetSize(getS), IPL_DEPTH_8U, 3);
		locations_x = new int[200];
		locations_y = new int[200];

		Mat fin;

		while (!inFile.eof()){
			inFile >> locations_x[i] >> locations_y[i];
			std::cout << locations_x[i] << locations_y[i] << endl;
			i++;
		}

		while (1){
			name = "Daegari";
			name +=std::to_string(t);
			name +=".jpg";
			std::cout << name << endl;
			input = imread(name);
			if (input.empty())
			{
				break;
			}

			std::cout << w << h<< endl;
			Rect rect(locations_x[t], locations_y[t], w, h);
			Mat subImage = bg(rect);
			//Step 1. mask만들기
			input.copyTo(gray);
			subImage.copyTo(graybg);
			temp2 = &IplImage(gray);
			cvNot(temp2, temp2);
			//Step 2. Min 함수 이용
			temp1 = &IplImage(graybg);
			cvCvtColor(temp2, trash2, CV_BGR2GRAY);
			cvThreshold(trash2, trash2, 250, 255, 0);
			Show("AAA", trash2);
			cvMerge(trash2, trash2, trash2, 0, merge);
			cvMin(merge,temp1, result);
			
			//Step3.Add
			//imshow("rr", subImage);
			fin = (Mat)result + input;
			//보여주고 쓰기
			imshow("HH", fin);
			name = "MO";	
			name += std::to_string(t);
			name += ".jpg";
			imwrite(name, fin);
			

			t++;
			if (waitKey(10) == 27 || t >= i+1) break;
		}

	}

	void Edge(){
		// Original Image

		VideoCapture vc("yjh.avi");
		Mat back = imread("Colored.jpg");
		Mat image;		
		Mat temp;

		double lowThresh = 100;
		double highThresh = 160;
		
		cvtColor(back, temp, CV_BGR2GRAY);
		Mat canny_back;

		Canny(temp, canny_back, lowThresh, highThresh);
		imshow("image2", canny_back);
		GaussianBlur(canny_back, canny_back, Size(11, 11), 2.0, 2.0);

		int i = 0;
		while (1){
			vc >> image;
			std::cout << "Got JV!"<< i++ << std::endl;

			if (image.empty()) break;
			//imshow("Original image", image);

			// Original Image to Gray Image
			Mat gray;
			cvtColor(image, gray, CV_BGR2GRAY);

			// Canny Filter
			Mat canny;

			Canny(gray, canny, lowThresh, highThresh);

			GaussianBlur(canny, canny, Size(11, 11), 2.0, 2.0);

			canny= canny- canny_back;;

			// Result Image
			imshow("image", canny);

			if (waitKey(10) == 27) break;
		}

	}
	void Person(){
		CvCapture *capture = cvCaptureFromFile("Example1.mp4");
		if (!capture) {
			std::cout << "The video file was not found.";
		}
		int width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		CvSize frameSize = cvSize(width, height);

		IplImage*image1;
		IplImage*image_sub = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage*image_sub2 = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage*image_mask= cvCreateImage(frameSize, 8, 1);
		IplImage*img2gray = cvCreateImage(frameSize, 8, 1);
		IplImage*img1_fg = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		int t = 0;
		image1 = cvLoadImage("Colored.jpg");
		std::cout << "Got Colored" << std::endl;

		IplImage *frame = NULL;

		static int first = 1;
		int Icount = 0;

		while (1) {

			frame = cvQueryFrame(capture);
			std::cout << frameSize.height << frameSize.width << std::endl;
			std::cout << "Got Query" << std::endl;
			if (!frame)    break;
			t++;
			std::cout << t << std::endl;

			// cvSub 함수를 사용하여 배경제거

			cvSub(frame, image1, image_sub); 

			//cvNot(image_sub, image_sub);
			//cvCvtColor(image_sub, img2gray, CV_BGR2GRAY);
			//cvThreshold(img2gray, image_mask, 10, 255, THRESH_BINARY);
			cvThreshold(image_sub, image_sub, 10, 255, 1);

	//		cvAdd(image_sub, image_sub, img1_fg, image_mask);
			//cvCvtColor(image_sub, image_mask, CV_BGR2GRAY);
			

			//cvAddWeighted(frame,0.4, img1_fg,0.6,1, img1_fg);

			cvShowImage("Minus", image_sub);

			char chKey = cvWaitKey(10);
			if (chKey == 27)    // ESC    
				break;
		}



		cvDestroyAllWindows();
		cvReleaseImage(&image1);
		cvReleaseImage(&image_sub);
		cvReleaseCapture(&capture);
	}
	void Person2(){
		CvCapture *capture = cvCaptureFromFile("Example1.mp4");
		CvCapture *capture2 = cvCaptureFromFile("Example1.mp4");
		if (!capture) {
			std::cout << "The video file was not found.";
		}
		int width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		CvSize frameSize = cvSize(width, height);

		IplImage *image1;
		IplImage*image_sub = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		int t = 0;
		image1 = cvLoadImage("Colored.jpg");
		std::cout << "Got Colored" << std::endl;

		IplImage *frame = NULL;
		IplImage *I = NULL; 
		IplImage *Imaskr = cvCreateImage(frameSize, 8, 1);
		IplImage *Imaskg = cvCreateImage(frameSize, 8, 1);
		IplImage *Imaskb = cvCreateImage(frameSize, 8, 1);
		IplImage *Igray1 = cvCreateImage(frameSize, 8, 1);
		IplImage *Igray2 = cvCreateImage(frameSize, 8, 1);
		IplImage *Igray3 = cvCreateImage(frameSize, 8, 1);
		IplImage *two = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage *Iscratch = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage *IavgF = cvCreateImage(frameSize, 8, 3);
		IplImage *IprevF = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage *Iscratch2 = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage *IdiffF = cvCreateImage(frameSize, 8, 3);
		IplImage *IhiF = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage *Ihi1 = cvCreateImage(frameSize, 8, 1);
		IplImage *Ihi2 = cvCreateImage(frameSize, 8, 1);
		IplImage *Ihi3 = cvCreateImage(frameSize, 8, 1);
		IplImage *Ihi4 = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage *IlowF = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage *Ilow1 = cvCreateImage(frameSize, 8, 1);
		IplImage *Ilow2 = cvCreateImage(frameSize, 8, 1);
		IplImage *Ilow3 = cvCreateImage(frameSize, 8, 1);
		IplImage *Ilow4 = cvCreateImage(frameSize, 8, 1);

		static int first = 1;
		int Icount = 0;

		while (1) {
			// capture로부터 프레임을 획득하여 포인터 frame에 저장한다.
			frame = cvQueryFrame(capture);
			std::cout << "Got Query" << std::endl;
			if (!frame)    break;
			t++;
			std::cout << t << std::endl;
			
			cvConvertScale(frame, Iscratch, 1, 0); //8비트 3채널 -> float타입 3채널 변환

			if (!first) {
				cvAcc(Iscratch, IavgF);
				cvAbsDiff(frame, IprevF, IdiffF); //프레임간의 절대값 영상을 구함
				cvAcc(Iscratch2, IdiffF);               //프레임간의 절대값 누적
				Icount += 1.0;
			}

			first = 0;
			cvCopy(frame, IprevF);
		}
		cvConvertScale(IdiffF, IdiffF, (double)1.0 / Icount);  //프레임간 절대값 누적 평균
		cvConvertScale(IavgF, IavgF, (double)1.0 / Icount);

		//IdiffF 영상의 각 채널값은 0보다 커야한다.
		cvAddS(IdiffF, cvScalar(1.0, 1.0, 1.0), IdiffF);
		
		cvConvertScale(IdiffF, Iscratch, 7.0);
		cvAdd(Iscratch, image1, IhiF);
		cvSplit(image1, Ihi1, Ihi2, Ihi3, 0);
		
		cvConvertScale(IdiffF, Iscratch, 6.0);
		cvAdd(image1, Iscratch, IlowF);
		cvSplit(image1, Ilow1, Ilow2, Ilow3, 0);
		
		std::vector<cv::Mat> array_to_merge;
		while (1) {
			std::cout << "Got Query" << std::endl;
			I = cvQueryFrame(capture2);
			if (!I)    break;
			cvConvertScale(I, Iscratch, 1, 0);
			cvSplit(Iscratch, Igray1, Igray2, Igray3, 0);

			//채널1
			cvInRange(Igray1, Ilow1, Ihi1, Imaskr);
			
			//채널2
			cvInRange(Igray2, Ilow2, Ihi2, Imaskg);
			cvOr(Imaskr, Imaskg, Imaskr);

			//채널3
			cvInRange(Igray3, Ilow3, Ihi3, Imaskb);
			cvOr(Imaskr, Imaskb, Imaskr);


			Mat B(Imaskb);
			Mat G(Imaskg);
			Mat R(Imaskr);

			array_to_merge.push_back(B);
			array_to_merge.push_back(G);
			array_to_merge.push_back(R);

			cv::Mat color(frameSize, CV_8UC3);;
			cv::merge(array_to_merge, color);
			//imwrite("merged.jpg", color);


			// cvSub 함수를 사용하여 배경제거
			//cvAbsDiff(frame, image1, image_sub);
			//cvLaplace(image_sub, two, 3);
			Show("TT",Imaskr);
			array_to_merge.clear();

			char chKey = cvWaitKey(10);
			if (chKey == 27)    // ESC    
				break;
		}



		cvDestroyAllWindows();
		cvReleaseImage(&image1);
		cvReleaseImage(&image_sub);
		cvReleaseCapture(&capture);
	}
	void Person3(){
		CvCapture *capture = cvCaptureFromFile("Example1.mp4");
		if (!capture)    {
			std::cout << "The video file was not found" << std::endl;
		}

		int width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		CvSize size = cvSize(width, height);

		// Initial background image
		IplImage *bkgImage = cvLoadImage("Colored.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		cvShowImage("bkgImage", bkgImage);

		// cvCreateImage 함수를 사용하여 그레이 스케일로 변환을 위한 grayImage와 차영상을 위한 diffImage를 생성한다.
		IplImage *grayImage = cvCreateImage(size, IPL_DEPTH_8U, 1);
		IplImage *diffImage = cvCreateImage(size, IPL_DEPTH_8U, 1);
		IplImage *frame = NULL;

		int t = 0;    // frame count
		int nThreshold = 50;

		while (1)    {
			// capture로부터 프레임을 획득하여 포인터 frame에 저장한다.
			frame = cvQueryFrame(capture);
			if (!frame)    break;
			t++;
			std::cout << t << std::endl;

			// cvCvtColor 함수를 사용하여 cvQueryFrame 함수로 획득한 frame을 그레이 스케일 영상으로 변환하여 grayImage에 저장한다.
			cvCvtColor(frame, grayImage, CV_BGR2GRAY);

			// cvAbsDiff 함수로 현재의 입력 비디오 프레임과 그레이 스케일 영상인 grayImage와 배경 영상인 bkgImage와의 차이의 절대값을 계산하여 diffImage에 저장한다.
			cvAbsDiff(grayImage, bkgImage, diffImage);

			// diffImage에서 0인 화소는 변화가 없는 화소이며, 값이 크면 클수록 배경 영상과의 차이가 크게 일어난 화소이다.
			// cvThreshold 함수를 사용하여 cvThreshold=50 이상인 화소만을 255로 저장하고, 임계값 이하인 값은 0으로 저장한다.
			// 임계값은 실험 또는 자동으로 적절히 결정해야 한다.
			cvThreshold(diffImage, diffImage, nThreshold, 255, CV_THRESH_BINARY);

			cvShowImage("grayImage", grayImage);
			cvShowImage("diffImage", diffImage);

			char chKey = cvWaitKey(10);
			if (chKey == 27)    {    // Esc
				break;
			}
		}

		cvDestroyAllWindows();
		cvReleaseImage(&grayImage);
		cvReleaseImage(&diffImage);
		cvReleaseCapture(&capture);
	}
	void Person4(){
		CvCapture *capture = cvCaptureFromFile("Example1.mp4");

		int width = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int height = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		CvSize frameSize = cvSize(width, height);

		IplImage *image1;
		IplImage*image_sub = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage*image_tt = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage*image_r = cvCreateImage(frameSize, 8, 1);
		IplImage*image_g = cvCreateImage(frameSize, 8, 1);
		IplImage*image_b = cvCreateImage(frameSize, 8, 1);
		IplImage*image_rgb= cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		IplImage*image_grey = cvCreateImage(frameSize, 8, 1);
		IplImage*image_greylow = cvCreateImage(frameSize, 8, 1);
		IplImage*img1_fg = cvCreateImage(frameSize, IPL_DEPTH_8U, 3);
		int t = 0;
		image1 = cvLoadImage("Colored.jpg");
		std::cout << "Got Colored" << std::endl;

		IplImage *frame = NULL;

		static int first = 1;
		int Icount = 0;
		
		CvScalar r, g, b;

		while (1) {

			frame = cvQueryFrame(capture);
			std::cout << frameSize.height << frameSize.width << std::endl;
			std::cout << "Got Query" << std::endl;
			if (!frame)    break;
			t++;
			std::cout << t << std::endl;

			cvCvtColor(frame, image_grey, CV_BGR2GRAY);
			
			
			cvThreshold(image_grey, image_greylow, 37, 255, 1);
			cvThreshold(image_grey, image_grey, 37, 255, 3);
			cvAdd(image_greylow, image_grey, image_grey);

			cvCvtColor(image_grey, image_tt, CV_GRAY2BGR);
			cvSplit(image_tt, image_r, image_g, image_b, 0);


			
			// cvSub 함수를 사용하여 배경제거
			//cvSub(frame, image1, image_sub);

			//cvShowImage("Minus", image_tt);

			//rgb
			for (int y = 0; y<height; y++)           // 1/3 영역 img_b 생성한곳에 블루색 변경
			{
				for (int x = 0; x<width; x++)
				{
					CvScalar s = cvGet2D(image_b, y, x);
					b.val[0] = s.val[0];
					cvSet2D(image_b, y, x, b);
				}
			}
			for (int y = 0; y<height; y++)           // 1/3 영역 img_b 생성한곳에 블루색 변경
			{
				for (int x = 0; x<width; x++)
				{
					CvScalar s = cvGet2D(image_g, y, x);
					g.val[1] = s.val[1];
					cvSet2D(image_g, y, x, g);
				}
			}
			for (int y = 0; y<height; y++)           // 1/3 영역 img_b 생성한곳에 블루색 변경
			{
				for (int x = 0; x<width; x++)
				{
					CvScalar s = cvGet2D(image_r, y, x);
					r.val[2] = s.val[2];
					cvSet2D(image_r, y, x, r);
				}
			}
			for (int y = 0; y<height; y++)           // RGB 이미지 합치기
			{
				for (int x = 0; x<width; x++)
				{
					CvScalar blue = cvGet2D(image_b, y, x);
					CvScalar green = cvGet2D(image_g, y, x);
					CvScalar red = cvGet2D(image_r, y, x);

					CvScalar rgb;
					rgb.val[0] = blue.val[0];
					rgb.val[1] = green.val[1];
					rgb.val[2] = red.val[2];

					cvSet2D(image_rgb, y, x, rgb);
				}
			}
			cvShowImage("Minus", image_rgb);

			char chKey = cvWaitKey(10);
			if (chKey == 27)    // ESC    
				break;
		}



		cvDestroyAllWindows();
		cvReleaseImage(&image1);
		cvReleaseImage(&image_sub);
		cvReleaseCapture(&capture);

	}
	void Polyfitting(){
		ifstream inFile("output_pano.txt");
		IplImage *src = cvLoadImage("Art_Pano.jpg", -1);
		double x[300], y[300];
		int i = 0;
		while (!inFile.eof()){
			inFile >> x[i] >> y[i];
			std::cout << x[i] <<"  "<< y[i] << endl;
			std::cout << i << endl;
			i++;
		}
		int n = 3;
		double X[2 * 3+ 1];
		for (int k = 0; k < 2 * n + 1; ++k){
			X[k] = 0;
			for (int j = 0; j < i; ++j){
				X[k] = X[k] + pow(x[j], k);
			}
		}
		double B[3 + 1][3 + 2];
		double a[3 + 1];
		for (int k = 0; k <= n; ++k){
			for (int j = 0; j <= n; j++)
			{
				B[k][j] = X[k + j];
			}
		}
		double Y[3 + 1];
		for (int k = 0; k<n + 1; k++)
		{
			Y[k] = 0;
			for (int j = 0; j<i; j++)
				Y[k] = Y[k] + pow(x[j], k)*y[j];
		}
		for (int k = 0; k <= n; k++)
			B[k][n + 1] = Y[k];
		n = n + 1; 
		cout << "\nThe Normal(Augmented Matrix) is as follows:\n";
		for (int k = 0; k < n; k++){
			for (int j = 0; j <= n; j++){
				cout << B[k][j] << setw(16);
			}
			cout << "\n";
		}
		//Gaussian elemination
		for (int p = 0; p < n; p++)      {
			for (int k = p + 1; k < n; k++)
			{
				if (B[p][p] < B[k][p])
				for (int j = 0; j <= n; j++)
				{
					double temp = B[p][j];
					B[p][j] = B[k][j];
					B[k][j] = temp;
				}
			}
		}
		//Loop to do Gauss
		for (int p = 0; p < n - 1; p++){
			for (int k = p + 1; k < n; k++){
				double t = B[k][p] / B[p][p];
				for (int j = 0; j <= n; j++){
					B[k][j] = B[k][j] - t*B[p][j];
				}
			}
		}
		//Back substitution
		for (int p = n - 1; p >= 0; p--)             
		{                        //x is an array whose values correspond to the values of x,y,z..
			a[p] = B[p][n];                //make the variable to be calculated equal to the rhs of the last equation
			for (int j = 0; j<n; j++)
			if (j != p)            //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
				a[p] = a[p] - B[p][j] * a[j];
			a[p] = a[p] / B[p][p];            //now finally divide the rhs by the coefficient of the variable to be calculated
		}
		cout << "\nThe values of the coefficients are as follows:\n";
		for (int p = 0; p<n; p++)
			cout << "x^" << p << "=" << a[p] << endl;
		cout << "\nHence the fitted Polynomial is given by:\ny=";
		for (int t = 0; t<n; t++)
			cout << " + (" << a[t] << ")" << "x^" << t;
		cout << "\n";

		//Fitting
		ofstream outFile("output_pano2.txt");
		double tempx, tempy;
		for (int p = 0; p < i;++p){
			tempx = x[p];
			tempy = a[0] + a[1] * tempx + a[2] * tempx*tempx + a[3]; //* tempx*tempx*tempx + a[4] * tempx*tempx*tempx*tempx + a[5] * tempx* tempx*tempx*tempx*tempx;// + a[6] * tempx* tempx* tempx*tempx*tempx*tempx + a[7] * tempx* tempx* tempx* tempx*tempx*tempx*tempx;
			std::cout << tempx << "  " << tempy << endl;
			if (tempy>0) outFile << tempx << " " << tempy << endl;
			else outFile << tempx << " " << 0 << endl;
		}

		Show("imp",src);
		cvWaitKey(0);
	}
	void CurveFitting(){
		ifstream inFile("output.txt");
		vector<Point> contour;
		int x, y;
		int i = 1;
		while (!inFile.eof()){
			inFile >> x >> y;
			std::cout << x << y << endl;
			contour.push_back(Point2f(x, y));
		}
		vector<Point> approx;
		approxPolyDP(contour, approx, 5, false);
		ofstream outFile("output_fit.txt");
		int length = approx.size();
		for (int j = 0; j < length; j++){
			std::cout << approx.at(j)<< endl;
			outFile << approx.at(j) << endl;

		}
		outFile.close();
	}
	void MovingCam(){
		CvCapture *capture = cvCaptureFromFile("PanoP.mp4"); 
		int w = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int h = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		
		VideoCapture vc("PanoP.mp4");

		Mat img;
		//fps
		double fps = vc.get(CV_CAP_PROP_FPS);
		int fps_int = floor(fps + 0.5);
		int delay = 1000 / fps;
		int length = vc.get(CV_CAP_PROP_FRAME_COUNT);
		std::cout << fps << std::endl;
		std::cout << length << std::endl;
		locations_x = new int[length];
		locations_y = new int[length];

		//bg
		Mat bg = imread("stitched_Pano.jpg");
		Mat AST = imread("PanoPP.jpg");
		ifstream inFile("output_person.txt");
		int i = 0;

		Mat img_s;
		IplImage*image_sub = cvCreateImage(Size(w,h), IPL_DEPTH_8U, 3);
		IplImage*image_tt = cvCreateImage(Size(w, h), IPL_DEPTH_8U, 3);
		IplImage*image_momo = cvCreateImage(Size(w, h), IPL_DEPTH_8U, 3);
		IplImage*image_true = cvCreateImage(Size(w, h), IPL_DEPTH_8U, 3);
		IplImage*image_false = cvCreateImage(Size(w, h), IPL_DEPTH_8U, 3);
		IplImage*image_mask = cvCreateImage(Size(w, h), 8, 1);
		IplImage *frame = NULL;

		IplImage*back;

		while (!inFile.eof()){
			inFile >> locations_x[i] >> locations_y [i];
			std::cout << locations_x[i] << locations_y[i] << endl;
			i++;
		}
		int j = 0;
		while (1){
			clock_t startTime = clock();
			vc >> img;

			Rect rect(locations_x[j], locations_y[j], w, h);
			Mat subImage = bg(rect);
			Mat backImage = AST(rect);
			//GaussianBlur(subImage, subImage, Size(9, 9), 2.0, 2.0);
			GaussianBlur(img, img, Size(9, 9), 2.0, 2.0);

			img_s = (img - subImage);
			frame = &IplImage(img);
			back = &IplImage(backImage);

			image_sub = &IplImage(img_s);
			cvCvtColor(image_sub, image_mask, CV_BGR2GRAY);
			cvThreshold(image_mask, image_mask, 13, 255, 0);
			cvAnd(frame, frame, image_true, image_mask);
			cvMerge(image_mask, image_mask, image_mask, 0, image_tt);

			cvNot(image_mask, image_mask);
			cvAnd(back, back, image_false, image_mask);
			cvMerge(image_mask, image_mask, image_mask, 0, image_momo);
			cvMin(image_true, image_tt, image_true);
			cvMin(image_false, image_momo, image_false);
			
			cvAdd(image_true, image_false, image_true);
			Show("image", image_true);

			string name;
			name = "Pano";
			name += std::to_string(j);
			name += ".jpg";
			imwrite(name, (Mat)image_true);
			j++;
			
			while (clock() - startTime < delay) {
				waitKey(1);
			}
			if (waitKey(10) == 27||j==length) break;
		}
	}
	void CutImages(){
		CvCapture *capture = cvCaptureFromFile("NoP (3).mp4");
		int w = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
		int h = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
		Mat bg = imread("StopPano.jpg");
		ifstream inFile("output_pano2.txt");
		double x, y;
		int i = 1;
		while (!inFile.eof()){
			inFile >> x >> y;
			std::cout << x << y << endl;
			Rect rect((int)x, (int)y, w, h);
			Mat subImage = bg(rect);
			string name;
			name = "Aft";
			name += std::to_string(i);
			name += ".jpg";
			imwrite(name, subImage);
			i++;
			
		}
		imshow("TT", bg);
		cvWaitKey(0);
	}
	void VideoCut(){
		double min, max;
		CvPoint left_top;
		// 먼저, source image를 로드한 후,
		IplImage *src = cvLoadImage("stitched_Pano.jpg", -1);

		//Video load
		VideoCapture vc("PanoP.mp4");

		//FPS
		double fps = vc.get(CV_CAP_PROP_FPS);
		int fps_int = floor(fps + 0.5);
		int delay = 1000 / fps;
		int length = vc.get(CV_CAP_PROP_FRAME_COUNT);
		std::cout << fps << std::endl;
		std::cout << length << std::endl;
		locations_x = new int[length + 3];
		locations_y = new int[length + 3];

		int i = 0;
		Mat img;
		while (1){
			clock_t startTime = clock();

			vc >> img;
			if (img.empty()) break;
			imshow("video", img);
			// template image를 로드한다. 
			IplImage *temp = &IplImage(img);
			// 상관계수를 구할 이미지
			IplImage *coeff = cvCreateImage(cvSize(src->width - temp->width + 1, src->height - temp->height + 1), IPL_DEPTH_32F, 1);
			// 상관계수를 구하여 coeff에 그려준다.
			cvMatchTemplate(src, temp, coeff, CV_TM_CCOEFF_NORMED);
			// 상관계수가 최대값을 가지는 위치를 찾는다 
			cvMinMaxLoc(coeff, &min, &max, NULL, &left_top);
			// 찾은 물체에 사격형 박스를 그린다.
			cvRectangle(src, left_top, cvPoint(left_top.x + temp->width,
				left_top.y + temp->height), CV_RGB(255, 0, 0), 2);
			//상관계수 최대값
			cout << max << endl;

			std::cout << left_top.x << "  " << left_top.y << std::endl;
			locations_x[i] = left_top.x;
			locations_y[i] = left_top.y;
			std::cout << locations_x[i] << "  " << locations_y[i] << std::endl;
			++i;
			Show("Matching Result", src);  // 매칭 결과 이미지
			Show("Template", temp);        // 템플릿 이미지
			//Show("Coefficient", coeff);      // 상관계수 이미지 보기
			while (clock() - startTime < delay) {
				waitKey(1);
			}
			if (waitKey(10) == 27) break;
		}
		ofstream outFile("output_person.txt");

		for (int j = 0; j < length; j++){
			outFile << locations_x[j] << " " << locations_y[j] << endl;
		}
		outFile.close();
		std::cout << "the end" << std::endl;

		cvReleaseImage(&src);
		cvDestroyAllWindows();
	}
	void Resize(){
		Mat trans = imread("Bad.jpg");
		IplImage *ori = cvLoadImage("stitched_Noop3.jpg",-1);
		Mat dst;
		int new_w, new_h;
		new_w = ori->width;
		new_h = ori->height;

		int row = trans.rows;
		std::cout << new_h << std::endl;
		std::cout << row << std::endl;
		double perc = (double)new_h*1 / (double)row;
		std::cout << perc << std::endl;

		resize(trans, dst, Size(new_w,new_h));
		
		imwrite("StopPano.jpg", dst);
		imshow("free", dst);
		cvWaitKey(0);
		destroyAllWindows();
	}
	void Show(char *str, IplImage *img)
	{
		cvNamedWindow(str, 1);
		cvShowImage(str, img);
	}	
	void Template(){
		double min, max;
		CvPoint left_top;
		// 먼저, source image를 로드한 후,
		IplImage *src = cvLoadImage("stitched_Depth_2.jpg", -1);
		// template image를 로드한다. 
		IplImage *temp = cvLoadImage("(4)_org.jpg", -1);

		// 상관계수를 구할 이미지
		IplImage *coeff = cvCreateImage(cvSize(src->width - temp->width + 1, src->height - temp->height + 1), IPL_DEPTH_32F, 1);
		// 상관계수를 구하여 coeff에 그려준다.
		cvMatchTemplate(src, temp, coeff, CV_TM_CCOEFF_NORMED);
		// 상관계수가 최대값을 가지는 위치를 찾는다 
		cvMinMaxLoc(coeff, &min, &max, NULL, &left_top);
		// 찾은 물체에 사격형 박스를 그린다.
		cvRectangle(src, left_top, cvPoint(left_top.x + temp->width,
			left_top.y + temp->height), CV_RGB(255, 0, 0), 2);
		//상관계수 최대값
		cout << max<<endl;

		std::cout << left_top.x <<"  "<<left_top.y << std::endl;
		Show("Matching Result", src);  // 매칭 결과 이미지
		Show("Template", temp);        // 템플릿 이미지
		Show("Coefficient", coeff);      // 상관계수 이미지 보기
		cvWaitKey(0);

		cvReleaseImage(&src);
		cvReleaseImage(&temp);
		cvReleaseImage(&coeff);
		cvDestroyAllWindows();
	}
	void PicTemplate(){
		string name1;
		const char*name_1;
		int t = 1;


		double min, max;
		CvPoint left_top;
		// 먼저, source image를 로드한 후,
		IplImage *src = cvLoadImage("stitched_Noop3_1.jpg", -1);
		// template image를 로드한다. 
		IplImage *temp;
		IplImage *coeff;
		locations_x = new int[100];
		locations_y = new int[100];

		while (1){
			name1 = " (";
			name1 += std::to_string(t);
			name1 += ").jpg";
			name_1 = name1.c_str();
			
			temp = cvLoadImage(name_1, -1);
			if (!temp) break;
			// 상관계수를 구할 이미지
			coeff = cvCreateImage(cvSize(src->width - temp->width + 1, src->height - temp->height + 1), IPL_DEPTH_32F, 1);
			// 상관계수를 구하여 coeff에 그려준다.
			cvMatchTemplate(src, temp, coeff, CV_TM_CCOEFF_NORMED);
			// 상관계수가 최대값을 가지는 위치를 찾는다 
			cvMinMaxLoc(coeff, &min, &max, NULL, &left_top);
			// 찾은 물체에 사격형 박스를 그린다.
			cvRectangle(src, left_top, cvPoint(left_top.x + temp->width,
				left_top.y + temp->height), CV_RGB(255, 0, 0), 2);
			//상관계수 최대값
			cout << max << endl;

			std::cout << left_top.x << "  " << left_top.y << std::endl;
			locations_x[t] = left_top.x;
			locations_y[t] = left_top.y;
			Show("Matching Result", src);  // 매칭 결과 이미지
			Show("Template", temp);        // 템플릿 이미지
			//Show("Coefficient", coeff);      // 상관계수 이미지 보기
			t++;
			
			char chKey = cvWaitKey(10);
			if (chKey == 27)    // ESC    
				break;
		}
		ofstream outFile("output_D2.txt");

		for (int j = 1; j < t; j++){
			outFile << locations_x[j] << " " << locations_y[j] << endl;
		}
		outFile.close();

		cvReleaseImage(&src);
		cvReleaseImage(&temp);
		cvReleaseImage(&coeff);
		cvDestroyAllWindows();

	}
	void StitVid(){
		VideoCapture vc("PanoP.mp4");
		/*Stitcher 클래스 생성*/
		Stitcher s = Stitcher::createDefault();

		//FPS
		int fps = vc.get(CV_CAP_PROP_FPS);
		int delay = 1000 / fps;
		std::cout << fps<< std::endl;

		// Stitching 함수
		Mat img;
		static vector<Mat> VectorInput;
		Mat VectorCopy;
		Mat MatResult;
		int k = 0;
		int countF = 0;
		int init = 0;

		/*IplImage *ori = cvLoadImage("(1)_org.jpg", -1);
		int new_w, new_h;
		new_w = ori->width;
		new_h = ori->height;
		*/
		while (1){
			clock_t startTime = clock();
			if (k % 22== 0){
				if (init == 0){
					std::cout << "Fuck" << std::endl;
					vc >> img;
					if (img.empty()) break;
					//resize(img, img, Size(new_w, new_h));

					VectorInput.push_back(img);
					imshow("RR", img);
					k++;
					countF++;
					init++;
				}
				else {
					std::cout << "Fuck" << std::endl;
					std::cout << k << std::endl;
					vc >> img;
					std::cout << "1" << std::endl;
					if (img.empty()) break;
					//resize(img, img, Size(new_w, new_h));
					VectorCopy=VectorInput[0].clone();
					std::cout << "2" << std::endl;
					VectorInput.push_back(img);
					std::cout << "3" << std::endl;
					imshow("RR", img);
					s.stitch(VectorInput, MatResult);
					std::cout << "4" << std::endl;
					if (MatResult.size().height != 0){
						std::cout << "Yeah" << std::endl;
						VectorInput.clear();
						VectorInput.push_back(MatResult);
					    k++;
						countF++;
					}
					else{
						std::cout << "shit" << std::endl;
						VectorInput.clear();
						VectorInput.push_back(VectorCopy);
						k++;
						countF++;
					}
					MatResult.release();
					VectorCopy.release();
				}
			}
			else{
				std::cout << "Damn" << std::endl;
				vc >> img;
				k++;
				if (img.empty()) break;
			}
			if (img.empty()) break;
			/*while (clock() - startTime < delay) {
				waitKey(1);
			}*/
			if (waitKey(10) == 27) break;
		}	
		std::cout << countF << std::endl;
		
		imshow("stitch", VectorInput[0]);
		waitKey(0);

		imwrite("stitched_Noop4.jpg", VectorInput[0]);
		destroyAllWindows();
	}
	void AVI(){
		VideoCapture vc("Love.mp4");
		if (!vc.isOpened()) return; // 불러오기 실패
		else
			std::cout << "Hello World" << std::endl;
		int fpsNum = vc.get(CV_CAP_PROP_FPS);
		std::cout << fpsNum << std::endl;
		Mat img;
		while (1){
			vc >> img;
			if (img.empty()) break;
			imshow("video", img);
			if (waitKey(10) == 27) break; //ESC
		}

		destroyAllWindows();
	}
	void SaveAvi(){
		IplImage * frame = 0;
		CvCapture *capture = cvCaptureFromAVI("Love.mp4");


		cvNamedWindow("Video", 0);
		CvVideoWriter *writer = 0;
		int fps = 30;

		while (1) {
			cvGrabFrame(capture);
			frame = cvRetrieveFrame(capture);
			if (!frame) {
				break;
			}
			writer = cvCreateVideoWriter("ballpark.avi", -1, fps, cvGetSize(frame));
			cvShowImage("Video", frame);
			cvWriteFrame(writer, frame);
			char chKey = cvWaitKey(10);
			if (chKey == 27)    // ESC    
				break;
		}
		cvReleaseVideoWriter(&writer);
		cvReleaseCapture(&capture);
		cvDestroyWindow("Video");
	}
	void Stit() {
		/*Stitcher 클래스 생성*/
		Stitcher s = Stitcher::createDefault();

		// Stitching 함수

		Mat input, mask;
	
		int i = 1;
		string name;
		string name2;
		vector<Mat> VectorInput;
		Mat MatResult;
		Mat VectorCopy;
		while (1){
			name = "(";
			name += std::to_string(i);
			name += ")_org.jpg";
			name2 = "Daegari";
			name2 += std::to_string(i);
			name2 += ".jpg";
			input = imread(name, CV_LOAD_IMAGE_COLOR);
			mask = imread(name2, CV_LOAD_IMAGE_COLOR);
			input = input - mask;
			if (input.empty()) break;
			if (i == 1){
				VectorInput.push_back(input);
				//imshow("RR", input);
			}
			else{
				VectorCopy = VectorInput[0].clone();
				VectorInput.push_back(input);
				s.stitch(VectorInput, MatResult);
				//imshow("RR", input);
				VectorCopy = VectorInput[0].clone();
				s.stitch(VectorInput, MatResult);
				if (MatResult.size().height != 0){
					std::cout << "Yeah" << std::endl;
					VectorInput.clear();
					VectorInput.push_back(MatResult);
				}
				else{
					std::cout << "shit" << std::endl;
					VectorInput.clear();
					VectorInput.push_back(VectorCopy);
				}
				MatResult.release();
				VectorCopy.release();
			}
			i +=2;
			if (waitKey(10) == 27) break;
		}
		std::cout << "end" << std::endl;
		imshow("stitch", VectorInput[0]);
		imwrite("stitchedhh.jpg", VectorInput[0]);
		waitKey(0);

	}
	void matt(){
		Mat image = imread("pusan.jpg", 1);
		imshow("Dd", image);
		waitKey(0);
		imwrite("copy.jpg", image);
	}
};


int main()
{
	ChangSul p = ChangSul();
	int an;
//	p.Resize();
	p.CutImages();
	return 0;
}
