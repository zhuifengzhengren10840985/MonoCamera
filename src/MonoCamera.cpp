#include <cmath>
#include <string>
#include <fstream>
#include<sstream>
#include <iostream>
#include <ros/ros.h>
#include <exception>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/thread.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>  
#include <MonoCamera/object.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <geometry_msgs/PoseStamped.h>

using namespace std;
using namespace cv;

//图像的长度和宽度
int width = 640;
int height = 480;

//像素坐标
struct point2D
{
    point2D(int _x, int _y) : x(_x), y(_y){}

    int x;
    int y;
};

//sort函数的第三个参数的仿函数
bool comp1(const point2D &a, const point2D &b){
    return a.x < b.x;
}

//数字检测的滑动条
int low_H_white1 = 0, 
       high_H_white1 = 89, 

       low_H_white2 = 90, 
       high_H_white2 = 180, 

       low_S_white = 0, 
       high_S_white = 255, 

       low_V_white = 0,
       high_V_white = 255;

//////////////////////////////////////////////////////////////////////////////////数字滑动条///////////////////////////////////////////////////////////////////////////////
const cv::String NumberDetection = "Number Detection";

static void on_low_H1_thresh_trackbar1(int, void *)
{
    low_H_white1 = std::min(high_H_white1-1, low_H_white1);//防止最大值小于最小值
	cv::setTrackbarPos("Low H1", NumberDetection, low_H_white1);
}
static void on_high_H1_thresh_trackbar1(int, void *)
{
    high_H_white1 = std::max(high_H_white1, low_H_white1+1);
	setTrackbarPos("High H1", NumberDetection, high_H_white1);
}

static void on_low_H2_thresh_trackbar1(int, void *)
{
    low_H_white2 = std::min(high_H_white2-1, low_H_white2);//防止最大值小于最小值
	cv::setTrackbarPos("Low H2", NumberDetection, low_H_white2);
}
static void on_high_H2_thresh_trackbar1(int, void *)
{
    high_H_white2 = std::max(high_H_white2, low_H_white2+1);
	setTrackbarPos("High H2", NumberDetection, high_H_white2);
}

static void on_low_S_thresh_trackbar1(int, void *)
{
    low_S_white = std::min(high_S_white-1, low_S_white);
	setTrackbarPos("Low S",NumberDetection,low_S_white);
}
static void on_high_S_thresh_trackbar1(int, void *)
{
    high_S_white = std::max(high_S_white,low_S_white+1);
	setTrackbarPos("High S",NumberDetection,high_S_white);
}

static void on_low_V_thresh_trackbar1(int, void *)
{
    low_V_white = std::min(high_V_white-1, low_V_white);
	setTrackbarPos("Low V",NumberDetection,low_V_white);
}
static void on_high_V_thresh_trackbar1(int, void *)
{
    high_V_white = std::max(high_V_white,low_V_white+1);
	setTrackbarPos("High V",NumberDetection,high_V_white);
}

//边框检测的滑动条
int low_H_red1 = 0, 
       high_H_red1 = 89, 

       low_H_red2 = 90, 
       high_H_red2 = 180, 

       low_S_red = 0, 
       high_S_red = 255, 

       low_V_red = 0,
       high_V_red = 255;

//////////////////////////////////////////////////////////////////////////////////////边框滑动条///////////////////////////////////////////////////////////////////////////////
const cv::String SquareDetection = "Square Detection";
static void on_low_H1_thresh_trackbar2(int, void *)
{
    low_H_red1 = std::min(high_H_red1-1, low_H_red1);//防止最大值小于最小值
	cv::setTrackbarPos("Low H1", SquareDetection, low_H_red1);
}
static void on_high_H1_thresh_trackbar2(int, void *)
{
    high_H_red1 = std::max(high_H_red1, low_H_red1+1);
	setTrackbarPos("High H1", SquareDetection, high_H_red1);
}

static void on_low_H2_thresh_trackbar2(int, void *)
{
    low_H_red2 = std::min(high_H_red2-1, low_H_red2);
	cv::setTrackbarPos("Low H2", SquareDetection, low_H_red2);
}
static void on_high_H2_thresh_trackbar2(int, void *)
{
    high_H_red2 = std::max(high_H_red2, low_H_red2+1);
	setTrackbarPos("High H2", SquareDetection, high_H_red2);
}

static void on_low_S_thresh_trackbar2(int, void *)
{
    low_S_red = std::min(high_S_red-1, low_S_red);
	setTrackbarPos("Low S",SquareDetection,low_S_red);
}
static void on_high_S_thresh_trackbar2(int, void *)
{
    high_S_red = std::max(high_S_red,low_S_red+1);
	setTrackbarPos("High S",SquareDetection,high_S_red);
}

static void on_low_V_thresh_trackbar2(int, void *)
{
    low_V_red = std::min(high_V_red-1, low_V_red);
	setTrackbarPos("Low V",SquareDetection,low_V_red);
}
static void on_high_V_thresh_trackbar2(int, void *)
{
    high_V_red = std::max(high_V_red,low_V_red+1);
	setTrackbarPos("High V",SquareDetection,high_V_red);
}

//原始,灰度,hsv,颜色分割图像
cv::Mat image_raw, image_gray, image_hsv;
cv::Mat image_threshold_white1,image_threshold_white2, image_threshold_white;
cv::Mat image_threshold_red1, image_threshold_red2, image_threshold_red;
cv::Mat image_erode, image_dilate;
cv::Mat image_erode1, image_dilate1;

void colorSeg1(){
    //边框检测HSV阈值调节窗口
    namedWindow(SquareDetection, 0);
    resizeWindow(SquareDetection, 640, 480);

    createTrackbar("Low H1", SquareDetection, &low_H_red1, 89, on_low_H1_thresh_trackbar2);
    createTrackbar("High H1", SquareDetection, &high_H_red1, 89, on_high_H1_thresh_trackbar2);

    createTrackbar("Low H2", SquareDetection, &low_H_red2, 180, on_low_H2_thresh_trackbar2);
    createTrackbar("High H2", SquareDetection, &high_H_red2, 180, on_high_H2_thresh_trackbar2);

    createTrackbar("Low S", SquareDetection, &low_S_red, 255, on_low_S_thresh_trackbar2);
    createTrackbar("High S", SquareDetection, &high_S_red, 255, on_high_S_thresh_trackbar2);

    createTrackbar("Low V", SquareDetection, &low_V_red, 255, on_low_V_thresh_trackbar2);
    createTrackbar("High V", SquareDetection, &high_V_red, 255, on_high_V_thresh_trackbar2);
    
    //颜色分割出红色,得到二值图像image_threshold_red
    // cv::inRange(image_hsv, cv::Scalar(0, 43, 46), cv::Scalar(10, 255, 255), image_threshold_red1);
    // cv::inRange(image_hsv, cv::Scalar(156, 43, 46), cv::Scalar(180, 255, 255), image_threshold_red2);
    // cv::bitwise_or(image_threshold_red1, image_threshold_red2, image_threshold_red);

    cv::inRange(image_hsv, cv::Scalar(low_H_red1,low_S_red,low_V_red), cv::Scalar(high_H_red1,high_S_red,high_V_red), image_threshold_red1);
    cv::inRange(image_hsv, cv::Scalar(low_H_red2,low_S_red,low_V_red), cv::Scalar(high_H_red2,high_S_red,high_V_red), image_threshold_red2);
    cv::bitwise_or(image_threshold_red1, image_threshold_red2, image_threshold_red);
    
    //形态学运算，先腐蚀(erode)再膨胀(dilate)
    cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4));
    cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4));
    cv::erode(image_threshold_red, image_erode1, kernel1);
    // imshow("red_erode",image_erode1);
    cv::dilate(image_erode1, image_dilate1, kernel2);
    cv::imshow(SquareDetection, image_dilate1);
}

void colorSeg2(){
    //数字检测HSV阈值调节窗口
    namedWindow(NumberDetection, 0);
    resizeWindow(NumberDetection, 640, 480);

    createTrackbar("Low H1", NumberDetection, &low_H_white1, 89, on_low_H1_thresh_trackbar1);
    createTrackbar("High H1", NumberDetection, &high_H_white1, 89, on_high_H1_thresh_trackbar1);

    createTrackbar("Low H2", NumberDetection, &low_H_white2, 180, on_low_H2_thresh_trackbar1);
    createTrackbar("High H2", NumberDetection, &high_H_white2, 180, on_high_H2_thresh_trackbar1);

    createTrackbar("Low S", NumberDetection, &low_S_white, 255, on_low_S_thresh_trackbar1);
    createTrackbar("High S", NumberDetection, &high_S_white, 255, on_high_S_thresh_trackbar1);

    createTrackbar("Low V", NumberDetection, &low_V_white, 255, on_low_V_thresh_trackbar1);
    createTrackbar("High V", NumberDetection, &high_V_white, 255, on_high_V_thresh_trackbar1);

    // cv::inRange(image_hsv, cv::Scalar(0, 0, 221), cv::Scalar(180,30,255), image_threshold_white);

    cv::inRange(image_hsv, cv::Scalar(low_H_white1,low_S_white,low_V_white), cv::Scalar(high_H_white1,high_S_white,high_V_white), image_threshold_white1);
    cv::inRange(image_hsv, cv::Scalar(low_H_white2,low_S_white,low_V_white), cv::Scalar(high_H_white2,high_S_white,high_V_white), image_threshold_white2);
    cv::bitwise_or(image_threshold_white1, image_threshold_white2, image_threshold_white);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(4,4));
    cv::erode(image_threshold_white, image_erode, kernel);
    cv::dilate(image_erode, image_dilate, kernel);
    cv::imshow(NumberDetection, image_dilate);
}
 


std::vector<cv::Vec4i> hierarchy1;
std::vector<std::vector<cv::Point> > contours1;
std::vector<std::vector<cv::Point> > mContours;
std::vector<std::vector<cv::Point> > squares1;
std::vector<cv::Point> squareFour;//四个角点
//检测标志
bool isDetected = false;
//四个角点和中心点(Point包含u和v值)
cv::Point min_Point1, min_Point2, min_Point3, min_Point4, min_central;
void getFourPoint(){
    Mat image_square= Mat::zeros(image_raw.size(), CV_8UC3);

    //寻找轮廓
    findContours(image_dilate1, contours1, hierarchy1, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    // drawContours(image_square, contours1, -1, Scalar(255, 255,255));
    // imshow("Square", image_square);
    // cout << "00000" << endl;

    if(!contours1.empty()){
        // cout << "11111" << endl;

        //清空上次处理的轮廓
        mContours.clear();

        //滤除靠近图像边缘的或者面积较小的轮廓
        for(int i = 0; i < contours1.size(); i++){
            //面积较小的噪点
            if(contourArea(contours1[i]) / (width*height) < 0.007){
                continue;
            }
            //靠近图像边缘的轮廓
            Rect bound = boundingRect(contours1[i]);
            if(bound.x < 2 || bound.y < 2 || (bound.x + bound.width + 2) > width || (bound.y + bound.height + 2) > height){
                continue;
            }
            mContours.push_back(contours1[i]);
        }

        image_square = Scalar::all(0);
        drawContours(image_square, mContours, -1, Scalar(255, 255, 255), 1, 8);
        imshow("Filter Contours", image_square);
    }

    cout << "mContours.size() = " << mContours.size() << endl;
    squares1.clear();
    if(!mContours.empty()){
        //计算每个轮廓的面积
        std::list<int> len_list1;

        for (int i = 0; i < mContours.size(); i++){
            len_list1.push_back(cv::contourArea(mContours[i]));
        }
        //取面积最大的轮廓(外轮廓)
        if (len_list1.size() > 0){
            squares1.push_back(mContours[std::distance(len_list1.begin(), max_element(len_list1.begin(), len_list1.end()))]);
        }

        // //取内轮廓
        // std::vector<std::vector<cv::Point> > squares1;
        // if (len_list1.size() > 0){
        //     std::list<int>::iterator it = len_list1.begin();
        //     advance(it,std::distance(len_list1.begin(), max_element(len_list1.begin(), len_list1.end())));
        //     len_list1.erase(it);
        //     squares1.push_back(mContours[std::distance(len_list1.begin(), max_element(len_list1.begin(), len_list1.end()))]);
        // }
        // cout << "22222" << endl;

        // image_square = Scalar::all(0);
        // drawContours(image_square, squares1, -1, Scalar(255, 255, 255), 1, 8);
        // imshow("最大轮廓", image_square);
    }




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////角点检测///////////////////////////////////////////////////////////////////////////////////
//     std::vector<std::vector<cv::Point> > contourPoly(1);
//     if(!squares1.empty()){
//         double epsilon = 0.1*arcLength(squares1[0], true);
//         approxPolyDP(squares1[0], contourPoly[0], epsilon, true);
//         // drawContours(image_square, contourPoly, 0, Scalar(0, 255, 0), 2, 8);
//         // imshow("Square", image_square);
//         //凸包检测(可能检测出来不是4个点)
//         convexHull(contourPoly[0], squareFour, true, true);
//         cout << "拟合顶点个数：" << squareFour.size() << endl;
//     }

// // // // ///////////////////////////////////////////////////////////////////////////////对角点进行排序/////////////////////////////////////////////////////////////////////////////
// // // //顶点顺序
// // // //                           1 —— 2
// // // //                           /        / 
// // // //                        3 —— 4
// // // //
// // // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//     //首先判断是否有四个角点
//     if(squareFour.size() == 4){
//         //检测标志位设置为true
//         isDetected = true;
//         cout << "检测到矩形框! " << endl;

//         //四个顶点
//         vector<point2D> vertex;
//         for(int i =0; i < 4; i++){
//             point2D point_temp(squareFour[i].x, squareFour[i].y);
//             vertex.push_back(point_temp);
//         }

//         //将四个顶点按照x升序排列(前两个是1,3, 后两个是2,4)
//         sort(vertex.begin(), vertex.end(), comp1);
        
//         //左侧角点
//         vector<point2D> vertex13;
//         vertex13.push_back(vertex[0]);
//         vertex13.push_back(vertex[1]);

//         //区分角点1和3
//         if(vertex13[0].y < vertex13[1].y){
//             min_Point1.x = vertex13[0].x;
//             min_Point1.y = vertex13[0].y;
//             min_Point3.x = vertex13[1].x;
//             min_Point3.y = vertex13[1].y;
//         }
//         else{
//             min_Point1.x = vertex13[1].x;
//             min_Point1.y = vertex13[1].y;
//             min_Point3.x = vertex13[0].x;
//             min_Point3.y = vertex13[0].y;
//         }

//         //右侧角点
//         vector<point2D> vertex24;
//         vertex24.push_back(vertex[2]);
//         vertex24.push_back(vertex[3]);

//         //区分角点2和4
//         if(vertex24[0].y < vertex24[1].y){
//             min_Point2.x = vertex24[0].x;
//             min_Point2.y = vertex24[0].y;
//             min_Point4.x = vertex24[1].x;
//             min_Point4.y = vertex24[1].y;
//         }
//         else{
//             min_Point2.x = vertex24[1].x;
//             min_Point2.y = vertex24[1].y;
//             min_Point4.x = vertex24[0].x;
//             min_Point4.y = vertex24[0].y;
//         }

//         min_central.x = (min_Point1.x + min_Point2.x + min_Point3.x + min_Point4.x)/4;
//         min_central.y = (min_Point1.y + min_Point2.y + min_Point3.y + min_Point4.y)/4;

//         Mat image_raw_copy = image_raw;
//         circle(image_raw_copy, min_Point1 , 3, Scalar(0, 255, 0), 2, 8, 0);
//         circle(image_raw_copy, min_Point2 , 3, Scalar(0, 255, 0), 2, 8, 0);
//         circle(image_raw_copy, min_Point3 , 3, Scalar(0, 255, 0), 2, 8, 0);
//         circle(image_raw_copy, min_Point4 , 3, Scalar(0, 255, 0), 2, 8, 0);
//         circle(image_raw_copy, min_central , 1, Scalar(0, 255, 0), 2, 8, 0);

//         drawContours(image_raw_copy, contourPoly, 0, Scalar(255, 255, 0), 1, 8);
//         imshow("拟合的矩形框", image_raw_copy);
//     }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    cout << "squares1.size() = " << squares1.size() << endl;
    Mat image_raw_copy = image_raw;
    if(!squares1.empty()){
        //检测标志位设置为true
        isDetected = true;
        cout << "检测到红色矩形框! " << endl;

        std::vector<cv::Point> find_V = squares1[0];
        std::vector<cv::Point> Line1, Line2, Line3, Line4;//直线拟合数据点
        cv::Vec4f line_para1, line_para2, line_para3, line_para4;
        std::vector<double> find_distance1(find_V.size());
        std::vector<double> find_distance2(find_V.size());
        std::vector<double> find_distance3(find_V.size());
        std::vector<double> find_distance4(find_V.size());

        cv::Point find_Point;
        int Point_pos1=0, Point_pos2=0, Point_pos3=0, Point_pos4=0;
        for(int i = 0; i<find_V.size();i++){
            find_Point = find_V[i];
            find_distance1[i]=abs((find_Point.x + find_Point.y) / sqrt(2));
            find_distance2[i]=abs((find_Point.x - find_Point.y + height) / sqrt(2));
            find_distance3[i]=abs((find_Point.x - find_Point.y - width) / sqrt(2));
            find_distance4[i]=abs((find_Point.x + find_Point.y - height - width) / sqrt(2));
        }
        if(find_distance1.size()>0&&find_distance2.size()>0&&find_distance3.size()>0&&find_distance4.size()>0){
            Point_pos1=std::distance(find_distance1.begin(), min_element(find_distance1.begin(), find_distance1.end()));
            min_Point1=find_V[Point_pos1];
            Point_pos2=std::distance(find_distance2.begin(), min_element(find_distance2.begin(), find_distance2.end()));
            min_Point2=find_V[Point_pos2];
            Point_pos3=std::distance(find_distance3.begin(), min_element(find_distance3.begin(), find_distance3.end()));
            min_Point3=find_V[Point_pos3];
            Point_pos4=std::distance(find_distance4.begin(), min_element(find_distance4.begin(), find_distance4.end()));
            min_Point4=find_V[Point_pos4];

            min_central.x = (min_Point1.x + min_Point2.x + min_Point3.x + min_Point4.x)/4;
            min_central.y = (min_Point1.y + min_Point2.y + min_Point3.y + min_Point4.y)/4;

            circle(image_raw_copy, min_Point1 , 3, Scalar(0, 255, 0), 2, 8, 0);
            circle(image_raw_copy, min_Point2 , 3, Scalar(0, 255, 0), 2, 8, 0);
            circle(image_raw_copy, min_Point3 , 3, Scalar(0, 255, 0), 2, 8, 0);
            circle(image_raw_copy, min_Point4 , 3, Scalar(0, 255, 0), 2, 8, 0);
            circle(image_raw_copy, min_central , 1, Scalar(0, 255, 0), 2, 8, 0);

            cv::line(image_raw_copy,min_Point1,min_Point2,cv::Scalar(0,255,0),1,8,0);
            cv::line(image_raw_copy,min_Point2,min_Point4,cv::Scalar(0,255,0),1,8,0);
            cv::line(image_raw_copy,min_Point3,min_Point4,cv::Scalar(0,255,0),1,8,0);
            cv::line(image_raw_copy,min_Point3,min_Point1,cv::Scalar(0,255,0),1,8,0);
        }
    }
    else if(squares1.empty()){
        isDetected = false;
        min_Point1.x = -1;
        min_Point1.y = -1;
        min_Point2.x = -1;
        min_Point2.y = -1;
        min_Point3.x = -1;
        min_Point3.y = -1;
        min_Point4.x = -1;
        min_Point4.y = -1;
        min_central.x = -1;
        min_central.y = -1;
        cout << "未检测到红色矩形框" << endl;
    }
    imshow("拟合的矩形框", image_raw_copy);
}





//////////////////////////////////////////////////////////////////////////////主函数/////////////////////////////////////////////////////////////////////////////////
int main(){
    //启动编号为0的相机
    cv::VideoCapture cap(0);
    //设置图像的长宽和帧率
    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480); 

    //图像帧率fps 30
    double fps = cap.get(CAP_PROP_FPS);
    //图像尺寸size 640 x 480
    Size size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));

    //创建KCF跟踪器
    Ptr<TrackerKCF> KCF_tracker = TrackerKCF::create();

    if(!cap.isOpened())
    {
        std::cout << "相机打开失败！" << std::endl;
        return -1;
    }

    while(1){
        char key = waitKey(1);
        //按q键退出
        if(key == 'q'){
            break;
        }

        cap >> image_raw;
        if(image_raw.empty()){
            break;
        }

        //颜色空间转换
        cv::cvtColor(image_raw, image_hsv, cv::COLOR_BGR2HSV);  
        //颜色分割
        colorSeg1();

        getFourPoint();

        // Rect boundingBox(m);
        // KCF_tracker->init(image_raw, boundingBox);
        
        waitKey(1);
    }
    cap.release();
    return 0;
}