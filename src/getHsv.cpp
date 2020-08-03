// //////////////////////////用鼠标获取图片上某一点的颜色HSV值//////////////////////////
#include <opencv2/opencv.hpp>
#include <iostream>
 
using namespace cv;
using namespace std;
 
#define WINDOW_NAME "【效果图窗口】"        //为窗口标题定义的宏 
 
void on_MouseHandle(int event, int x, int y, int flags, void* param);
int main()
{
    cv::VideoCapture cap(0);
    while(1)
    {
	int multiple = 1;//图片的放大倍数
	Mat inputImage;
    cap >> inputImage;
	Mat outputImage;
	resize(inputImage, inputImage, Size(multiple  * inputImage.cols, multiple  * inputImage.rows));
	cvtColor(inputImage, outputImage, COLOR_BGR2HSV);
 
	//设置鼠标操作回调函数
	namedWindow(WINDOW_NAME);
	setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&outputImage);
	imshow(WINDOW_NAME, inputImage);
    waitKey(1);
    }
    cap.release();
    return 0;
}
 
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
 
	Mat& image = *(cv::Mat*) param;
	switch (event)
	{
	//左键按下消息
	case EVENT_LBUTTONDOWN:
	{
 
		cout << static_cast<int>( image.at<Vec3b>(y, x)[0]) << ",";
		cout << static_cast<int>(image.at<Vec3b>(y, x)[1]) << ",";
		cout << static_cast<int>(image.at<Vec3b>(y, x)[2]) << endl;
	}
	break;
    }
}