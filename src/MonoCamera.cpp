#include <cmath>
#include <cstring>
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

string path = "/home/lxl/catkin_ws/src/MonoCamera/";

//图像的长度和宽度
int width = 640;
int height = 480;

//模板数量
int numTemp = 159;
int conTemp = 136;

//轮廓模板集
vector< vector<Point> > mContoursTemp; 

//数字
int match_number;

//检测标志
bool isDetected = false;

//四个角点和中心点(Point包含u和v值)
cv::Point min_Point1, min_Point2, min_Point3, min_Point4, min_central;

//Jacobi
#define J_N 4
//#define Epsilon 35
#define Max 15
const int N2 = 2 * J_N;
//深度值
double depth[J_N] = {300,300,300,300};
//牛顿迭代的初值(深度Z的初值)
double x_start[J_N] = {300,300,300,300};
//四个角点和中心的相机坐标
Eigen::Vector3d Pc1,Pc2,Pc3, Pc4, Pc_central;


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

// //////////////////////////////////////////////////////////////////////////////////数字滑动条///////////////////////////////////////////////////////////////////////////////
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

///////////////////////////////////////////////////////////////////////////////////////////模板匹配//////////////////////////////////////////////////////////////////////////////////
cv::Mat substract(cv::Mat &src, cv::Mat&dst){
	int sum = 0; 
    cv::Mat result = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; ++i){
		uchar *data1 = src.ptr<uchar>(i);
		uchar*data2 = dst.ptr<uchar>(i);
		uchar*data3 = result.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
			data3[j] = abs(data1[j] - data2[j]);
	}
	return result;
}

int get_One(cv::Mat image){
	int sum_one = 0;
	for (int i = 0; i < image.rows; ++i){
		uchar*data = image.ptr<uchar>(i);
		for (int j = 0; j < image.cols; ++j){
			if (data[j] > 0) 
                sum_one += 1; 
		}
	}
    return sum_one;
}

int temple_image(cv::Mat img)
{
	cv::Mat src_image; 
    int temp = 0; 
    float sum = 0; 
    float max = 200000000; 
    int seriesNum = 0;
	float c = 0.0; 
    float n1 = 0.0; 
    float n2 = 0.0; 
    cv::Mat result;
	for (int i = 0; i < numTemp; ++i){
		char name[520];
        std::sprintf(name, "/home/lxl/catkin_ws/src/MonoCamera/template/%d.jpg", i);
		src_image = cv::imread(name);
		if (!src_image.data) 
            std::cout << "数字模板库读取失败！" << std::endl; 
		result = substract(img, src_image);
		c = get_One(result);
		if (c < max) {
			max = c; 
            seriesNum = i;
		}
	}
	return seriesNum;
}

//////////////////////////////////////////////////////////////////////////////////////////在图片中标注数字//////////////////////////////////////////////////////////////////////////////////
void drawNumber(Mat &img, int num){
    //如果未识别到任何数字
    if(num < 0){
        return;
    }

    stringstream st;
    st << num;
    string text = st.str();
    putText(img, text, Point(5, 100), FONT_HERSHEY_SIMPLEX, 4, Scalar(0, 255, 0), 5);
}

/////////////////////////////////////////////////////////////////////////////////////////模板图片轮廓初始化/////////////////////////////////////////////////////////////////
void init_hand_template()
{
    Mat srcImage;
    Mat dstImage;
    vector< vector<Point> > tmpContours;
    vector< Vec4i > mHierarchy;

    vector<vector<Point> > tempSquares;
    list<int> tmp_area;

    mContoursTemp.clear();

    for(int i = 0; i < conTemp; i++)
    {
        tempSquares.clear();
        tmp_area.clear();

        char tmp_names[500];
        sprintf(tmp_names, "/home/lxl/catkin_ws/src/MonoCamera/image_dilate/%d.jpg", i);
        // 读取文件
        srcImage = imread(tmp_names, IMREAD_GRAYSCALE);

        // imshow("srcImage", srcImage);
        if(!srcImage.empty())
        {
            dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);

            // 寻找轮廓tmpContours(包含干扰)
            findContours(srcImage, tmpContours, mHierarchy, RETR_CCOMP, CHAIN_APPROX_NONE, Point(0, 0));

            if(tmpContours.size() > 0){
                //计算每个轮廓的面积tmp_area
                for (int i = 0; i < tmpContours.size(); i++){
                    tmp_area.push_back(cv::contourArea(tmpContours[i]));
                }

                //寻找面积最大的轮廓tmpContours
                if (tmp_area.size() > 0){
                    tempSquares.push_back(tmpContours[std::distance(tmp_area.begin(), max_element(tmp_area.begin(), tmp_area.end()))]);
                }

                mContoursTemp.push_back( tempSquares[0]);
            }
        }
        else{
            cout << "模板图像读取失败！" << tmp_names <<endl;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////将数据记录到txt文件中//////////////////////////////////////////////////////////////////////////
void writeData(){
    //在当前目录下新建一个txt文件
    ofstream outfile;
    //打开文件
    // outfile.open("/home/lxl/catkin_ws/src/MonoCamera/ObjPos.txt", std::ios::app);
    outfile.open(path + "ObjPos.txt", std::ios::app);

    //如果读取txt文件失败，就打印提示信息
    if(!outfile){
        std::cout << "TXT文件读取失败！" << endl;
    }
    
    //将数据写入txt文件
    outfile << isDetected << "\t" << match_number << "\t" << Pc_central[0] << "\t" << Pc_central[1] << "\t" << Pc_central[2] << "\n";

    //关闭文件
    outfile.close();
}

// /////////////////////////////////////////////////////////////////////////////////////////主函数///////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    ros::init(argc, argv, "object_publisher");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<MonoCamera::object>("object_pub",1);
    ros::Rate loop_rate(30);

    //获取模板轮廓
    init_hand_template();

    //启动编号为0的相机
    cv::VideoCapture cap(0);
    //设置图像的长宽和帧率
    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480); 
 
    //图像尺寸size 640 x 480
    Size size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
    //图像帧率fps 30
    double fps = cap.get(CAP_PROP_FPS);
    
    cv::VideoWriter writer(path + "video.avi", CV_FOURCC('M', 'P', '4', '2'), fps , size, true);


    if(!cap.isOpened())
    {
        std::cout << "相机打开失败！" << std::endl;
        return -1;
    }

///////////////////////////////////////////////跟踪/////////////////////////////////////////////////
    // namedWindow("output", WINDOW_AUTOSIZE);
    // //创建KCF跟踪器
    // Ptr<TrackerKCF> tracker = TrackerKCF::create();
    // cap.read(image_raw);
    // Rect2d boundingBox;
    // boundingBox = selectROI("output", image_raw);
    // if(boundingBox.width == 0 || boundingBox.height == 0){
    //     return -1;
    // }
    // //跟踪
    // tracker->init(image_raw, boundingBox);
    // while(cap.read(image_raw)){
    //     if(image_raw.empty()){
    //         return -1;
    //     }
    //     tracker->update(image_raw, boundingBox);
    //     rectangle(image_raw, boundingBox, Scalar(0, 0, 255), 2, 8, 0);
    //     imshow("跟踪", image_raw);
    //     //按c退出
    //     char c = waitKey(1);
    //     if(c == 27){
    //         break;
    //     }
    // }
//////////////////////////////////////////////////////////////////////////////////////////////////

    //原始,灰度,hsv,颜色分割图像
    cv::Mat image_raw, image_hsv;
    cv::Mat image_threshold_white1,image_threshold_white2, image_threshold_white;
    cv::Mat image_threshold_red1, image_threshold_red2, image_threshold_red;
    cv::Mat image_erode, image_dilate;
    cv::Mat image_erode1, image_dilate1;

    //门框
    std::vector<cv::Vec4i> hierarchy1;
    std::vector<std::vector<cv::Point> > contours1;
    std::vector<std::vector<cv::Point> > mContours;
    std::vector<std::vector<cv::Point> > squares1;
    std::vector<cv::Point> squareFour;//四个角点

    //数字
    cv::Mat  mask, image_roi;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<std::vector<cv::Point> > numContours;
    std::vector<std::vector<cv::Point> > numSquares;
    std::vector<cv::Vec4i> hierarchy;

    cap.read(image_raw);
    while(cap.read(image_raw)){
        char key = waitKey(1);
        //按q键退出
        if(key == 'q'){
            break;
        }

        //颜色空间转换
        cv::cvtColor(image_raw, image_hsv, cv::COLOR_BGR2HSV);  

        //////////////////////////////////////////////////////////////////////////////颜色分割出边框/////////////////////////////////////////////////////////////////////////////////
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

        ///////////////////////////////////////////////////////////////////////////获取边框的四个角点//////////////////////////////////////////////////////////////////////////////////
        Mat image_square= Mat::zeros(image_raw.size(), CV_8UC3);
        Mat image_raw_copy = image_raw;

        //寻找轮廓
        findContours(image_dilate1, contours1, hierarchy1, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

        if(!contours1.empty()){
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
                if(bound.x < 2 || bound.y < 2 || (bound.x + bound.width + 2) > image_raw.cols || (bound.y + bound.height + 2) > image_raw.rows){
                    continue;
                }
                mContours.push_back(contours1[i]);
            }

            image_square = Scalar::all(0);
            drawContours(image_square, mContours, -1, Scalar(255, 255, 255), 1, 8);
            imshow("Filter Contours", image_square);
        }

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
        }

        if(!squares1.empty()){
            //检测标志位设置为true
            isDetected = true;
            cout << "### 检测到目标 ### " << endl;

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
                min_Point3=find_V[Point_pos2];
                Point_pos3=std::distance(find_distance3.begin(), min_element(find_distance3.begin(), find_distance3.end()));
                min_Point2=find_V[Point_pos3];
                Point_pos4=std::distance(find_distance4.begin(), min_element(find_distance4.begin(), find_distance4.end()));
                min_Point4=find_V[Point_pos4];

                min_central.x = (min_Point1.x + min_Point2.x + min_Point3.x + min_Point4.x)/4;
                min_central.y = (min_Point1.y + min_Point2.y + min_Point3.y + min_Point4.y)/4;

                // cout << "像素坐标:" << endl;
                // cout << "point1:(" << min_Point1.x << ", " << min_Point1.y << ")" << endl;
                // cout << "point2:(" << min_Point2.x << ", " << min_Point2.y << ")" << endl;
                // cout << "point3:(" << min_Point3.x << ", " << min_Point3.y << ")" << endl;
                // cout << "point4:(" << min_Point4.x << ", " << min_Point4.y << ")" << endl;

                //      1——2
                //       |         |
                //       3——4
                circle(image_raw_copy, min_Point1 , 2, Scalar(0, 255, 0), 2, 8, 0);
                circle(image_raw_copy, min_Point2 , 2, Scalar(0, 255, 0), 2, 8, 0);
                circle(image_raw_copy, min_Point3 , 2, Scalar(0, 255, 0), 2, 8, 0);
                circle(image_raw_copy, min_Point4 , 2, Scalar(0, 255, 0), 2, 8, 0);
                circle(image_raw_copy, min_central , 2, Scalar(0, 255, 0), 2, 8, 0);
                //point 1 2构成line1，顺时针依次为line 2 3 4
                cv::line(image_raw_copy,min_Point1,min_Point2,cv::Scalar(0,255,0),1,8,0);//1——2
                cv::line(image_raw_copy,min_Point2,min_Point4,cv::Scalar(0,255,0),1,8,0);//2——4
                cv::line(image_raw_copy,min_Point3,min_Point4,cv::Scalar(0,255,0),1,8,0);//3——4
                cv::line(image_raw_copy,min_Point3,min_Point1,cv::Scalar(0,255,0),1,8,0);//3——1
            }
            ///////////////////////////////////////////////////////////////////////////解算目标的位置//////////////////////////////////////////////////////////////////////////////
            //像素坐标的齐次坐标(u, v, 1)
            Eigen::Vector3d min_P1,  min_P2,  min_P3 ,  min_P4;
            min_P1 << min_Point1.x, min_Point1.y, 1;
            min_P2 << min_Point2.x, min_Point2.y, 1;
            min_P3 << min_Point3.x, min_Point3.y, 1;
            min_P4 << min_Point4.x, min_Point4.y, 1;

            //相机的内参矩阵(3x3)
            Eigen::Matrix3d CameraM;
            Eigen::Vector4d CameraPram;
            CameraPram[0] = 366.997006;//fx
            CameraPram[1] = 366.447492;//fy
            CameraPram[2] = 331.133147;//cx
            CameraPram[3] = 239.626159;//cy
            CameraM << CameraPram[0], 0, CameraPram[2], 0, CameraPram[1], CameraPram[3], 0, 0, 1;

            //相机归一化坐标（Xc/Zc, Yc/Zc)T =  相机内参矩阵的逆 *(u, v)T
            Eigen::Vector3d trans_P1d, trans_P2d, trans_P3d, trans_P4d;
            trans_P1d = CameraM.inverse() * min_P1;
            trans_P2d = CameraM.inverse() * min_P2;
            trans_P3d = CameraM.inverse() * min_P3;
            trans_P4d = CameraM.inverse() * min_P4;

            //牛顿迭代求解三元非线性方程组解算深度信息
            float Epsilon =0.01;	
            Eigen::MatrixXd X1_Jac(1,4); 

            X1_Jac <<10,10,10,10;

            Eigen::MatrixXd Y0_Jac(1,7);

            Eigen::MatrixXd X0_Jac(1,4);
            double x_start[4];
            X0_Jac << x_start[0], x_start[1], x_start[2], x_start[3];
            Eigen::MatrixXd W_Jac(1,4);
            W_Jac<<100,100,100,100;
        
            Eigen::MatrixXd YY_Jac(1,4);
            Eigen::MatrixXd Y_temp_Jac(4,7);
            Eigen::MatrixXd jacobian_temp(4,7);
            Eigen::MatrixXd jacobian_temp_M(4,7);
            Eigen::MatrixXd Z_Jac(1,4);
            Eigen::MatrixXd	JAC(4,7);//jacobin矩阵

            double y0[J_N+3]={0,0,0,0,0,0,0}, jacobian[J_N][J_N+3], invjacobian[J_N+3][J_N], x1[J_N]={0,0,0,0}, errornorm, v[12], L=0;
            double D1 = 69, D2 = 71, D3 = 99, D4 = 69;//外轮廓实际大小(单位cm)
            int i_inv=0, j_inv=0, k_inv=0, i_jcb=0, iter = 0;

            v[0] = trans_P1d[0];//Xc1
            v[1] = trans_P2d[0];//Xc2
            v[2] = trans_P3d[0];//Xc3
            v[3] = trans_P4d[0];//Xc4
            v[4] = trans_P1d[1];//Yc1
            v[5] = trans_P2d[1];//Yc2
            v[6] = trans_P3d[1];//Yc3
            v[7] = trans_P4d[1];//Yc4
            v[8] = D1;
            v[9] = D2;
            v[10] = D3;
            v[11] = D4;

            do{
                iter = iter + 1;
                //计算雅克比矩阵 jacobian
                y0[0] = (v[0]* v[0]+ v[4]* v[4]+ 1)*depth[0]* depth[0] - 2 * (v[0] * v[1] +v[4] *v[5] +1)*depth[0] * depth[1] + (v[1] * v[1] + v[5] * v[5] + 1)*depth[1] * depth[1] - v[8] * v[8];
                y0[1] = (v[0]* v[0]+ v[4]* v[4]+ 1)*depth[0]* depth[0] - 2 * (v[0] * v[2] +v[4] *v[6] +1)*depth[0] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] * depth[2] - v[9] * v[9];
                y0[2] = (v[1] *v[1]+ v[5] *v[5]+ 1)*depth[1]* depth[1] - 2 * (v[1] * v[2] +v[5] *v[6] +1)*depth[1] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] *depth[2] - v[10] * v[10];
                y0[3] = (v[3]* v[3]+ v[7]* v[7]+ 1)*depth[3]* depth[3] - 2 * (v[3] * v[2] +v[7] *v[6] +1)*depth[3] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] *depth[2] - v[11] * v[11];
                y0[4] = (v[0]*depth[0]-v[1]*depth[1])*(v[6]*depth[2]-v[7]*depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(v[4]*depth[0]-v[5]*depth[1]);
                y0[5] = (v[0]*depth[0]-v[1]*depth[1])*(depth[2]-depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(depth[0]-depth[1]);
                y0[6] = (v[4]*depth[0]-v[5]*depth[1])*(depth[2]-depth[3])-(v[6]*depth[2]-v[7]*depth[3])*(depth[0]-depth[1]);
                //计算初值位置的值
                //计算雅克比矩阵的逆矩阵 invjacobian
                //jacobian have n*n element  计算函数雅克比的值
                JAC(0,0)= 2 * (v[0] * v[0] + v[4] * v[4] + 1)*depth[0] - 2 * (v[0] * v[1] + v[4] * v[5] + 1)*depth[1];
                JAC(0,1) = 2 * (v[0] * v[0] + v[4] * v[4] + 1)*depth[0] - 2 * (v[0] * v[2] + v[4] * v[6] + 1)*depth[2];
                JAC(0,2)= 0;
                JAC(0,3) = 0;
                JAC(0,4) =  (v[0])*(v[6]*depth[2]-v[7]*depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(v[4]); ;
                JAC(0,5) =  (v[0])*(depth[2]-depth[3])-(v[2]*depth[2]-v[3]*depth[3]);
                JAC(0,6) =  (v[4])*(depth[2]-depth[3])-(v[6]*depth[2]-v[7]*depth[3]);

                JAC(1,0) = -2 * (v[0] * v[1] + v[4] * v[5] + 1)*depth[0] + 2 * (v[1] * v[1] + v[5] * v[5] + 1)*depth[1];
                JAC(1,1) = 0;
                JAC(1,2) = 2 * (v[1] * v[1] + v[5] * v[5] + 1)*depth[1] - 2 * (v[1] * v[2] + v[5] * v[6] + 1)*depth[2];
                JAC(1,3) = 0;
                JAC(1,4) = (-v[1])*(v[6]*depth[2]-v[7]*depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(-v[5]);
                JAC(1,5) = (-v[1])*(depth[2]-depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(-1);
                JAC(1,6) = (-v[5])*(depth[2]-depth[3])-(v[6]*depth[2]-v[7]*depth[3])*(-1);

                JAC(2,0) = 0;
                JAC(2,1) = -2 * (v[0] * v[2] + v[4] * v[6] + 1)*depth[0] + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*depth[2];
                JAC(2,2) = -2 * (v[1] * v[2] + v[5] * v[6] + 1)*depth[1] + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*depth[2];
                JAC(2,3) = 2 * (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] - 2 * (v[3] * v[2] + v[7] * v[6] + 1)*depth[3];
                JAC(2,4) = (v[0]*depth[0]-v[1]*depth[1])*(v[6])-(v[2])*(v[4]*depth[0]-v[5]*depth[1]);
                JAC(2,5) = (v[0]*depth[0]-v[1]*depth[1])*(1)-(v[2]*(depth[0]-depth[1]));
                JAC(2,6) = (v[4]*depth[0]-v[5]*depth[1])*(1)-(v[6]*(depth[0]-depth[1]));

                JAC(3,0) = 0;
                JAC(3,1) = 0;
                JAC(3,2) = 0;
                JAC(3,3) = 2 * (v[3] * v[3] + v[7] * v[7] + 1) * depth[3] -2 * (v[3] * v[2] + v[7] * v[6] + 1) *depth[2];
                JAC(3,4) = (v[0]*depth[0]-v[1]*depth[1])*(-v[7])-(-v[3])*(v[4]*depth[0]-v[5]*depth[1]);
                JAC(3,5) = (v[0]*depth[0]-v[1]*depth[1])*(-1)-(-v[3])*(depth[0]-depth[1]);
                JAC(3,6) = (v[4]*depth[0]-v[5]*depth[1])*(-1)-(-v[7])*(depth[0]-depth[1]);

                Eigen::JacobiSVD<Eigen::MatrixXd> svd(JAC, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
                double  pinvtoler = 1.e-8; //tolerance
                int row_invJav = JAC.rows();
                int col_invJav= JAC.cols();
                int k_invJav =row_invJav>=col_invJav? col_invJav:row_invJav;
                Eigen::MatrixXd X_invJav = Eigen::MatrixXd::Zero(col_invJav,row_invJav);
                Eigen::MatrixXd singularValues_inv_invJav = svd.singularValues();//奇异值
                Eigen::MatrixXd singularValues_inv_mat_invJav = Eigen::MatrixXd::Zero(col_invJav, row_invJav);
                for (long i = 0; i<k_invJav; ++i) 
                {
                    if (singularValues_inv_invJav(i) > pinvtoler)
                        singularValues_inv_invJav(i) = 1.0 / singularValues_inv_invJav(i);
                    else 
                        singularValues_inv_invJav(i) = 0;
                }
                for (long i = 0; i < k_invJav; ++i) 
                {
                    singularValues_inv_mat_invJav(i, i) = singularValues_inv_invJav(i);
                }
                X_invJav=(svd.matrixV())*(singularValues_inv_mat_invJav)*(svd.matrixU().transpose());//X=VS+U*

                //优化版迭代部分
                Y0_Jac<<y0[0],y0[1],y0[2],y0[3],y0[4],y0[5],y0[6];
            
                X1_Jac = X0_Jac-Y0_Jac*X_invJav;//x1=depth-f/df

                jacobian_temp(0,0) = 2 * (v[0] * v[0] + v[4] * v[4] + 1)*X1_Jac(0,0) - 2 * (v[0] * v[1] + v[4] * v[5] + 1)*X1_Jac(0,1);
                jacobian_temp(0,1)  = 2 * (v[0] * v[0] + v[4] * v[4] + 1)*X1_Jac(0,0) - 2 * (v[0] * v[2] + v[4] * v[6] + 1)*X1_Jac(0,2);
                jacobian_temp(0,2)  = 0;
                jacobian_temp(0,3)  = 0;
                jacobian_temp(0,4)  =  (v[0])*(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3))*(v[4]); ;
                jacobian_temp(0,5)  =  (v[0])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3));
                jacobian_temp(0,6)  =  (v[4])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3));

                jacobian_temp(1,0)  = -2 * (v[0] * v[1] + v[4] * v[5] + 1)*X1_Jac(0,0) + 2 * (v[1] * v[1] + v[5] * v[5] + 1)*X1_Jac(0,1);
                jacobian_temp(1,1) = 0;
                jacobian_temp(1,2) = 2 * (v[1] * v[1] + v[5] * v[5] + 1)*X1_Jac(0,1) - 2 * (v[1] * v[2] + v[5] * v[6] + 1)*X1_Jac(0,2);
                jacobian_temp(1,3) = 0;
                jacobian_temp(1,4) = (-v[1])*(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3))*(-v[5]);
                jacobian_temp(1,5) = (-v[1])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[2]*X1_Jac(0,2)-v[3]*X1_Jac(0,3))*(-1);
                jacobian_temp(1,6) = (-v[5])*(X1_Jac(0,2)-X1_Jac(0,3))-(v[6]*X1_Jac(0,2)-v[7]*X1_Jac(0,3))*(-1);

                jacobian_temp(2,0) = 0;
                jacobian_temp(2,1) = -2 * (v[0] * v[2] + v[4] * v[6] + 1)*X1_Jac(0,0) + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*X1_Jac(0,2);

                jacobian_temp(2,2) = -2 * (v[1] * v[2] + v[5] * v[6] + 1)*X1_Jac(0,1) + 2 * (v[2] * v[2] + v[6] * v[6] + 1)*X1_Jac(0,2);
                jacobian_temp(2,3)  = 2 * (v[2] * v[2] + v[6] * v[6] + 1)*X1_Jac(0,2) - 2 * (v[3] * v[2] + v[7] * v[6] + 1) *X1_Jac(0,3);
                jacobian_temp(2,4) = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))*(v[6])-(v[2])*(v[4]*X1_Jac(0,0)-v[5]*X1_Jac(0,1));
                jacobian_temp(2,5) = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))-(v[2])*(X1_Jac(0,0)-X1_Jac(0,1));
                jacobian_temp(2,6) = (v[4]*X1_Jac(0,0)-v[5]*X1_Jac(0,1))*(1)-(v[6])*(X1_Jac(0,0)-X1_Jac(0,1));

                jacobian_temp(3,0)  = 0;
                jacobian_temp(3,1)  = 0;
                jacobian_temp(3,2) = 0;
                jacobian_temp(3,3)  = 2 * (v[3] * v[3] + v[7] * v[7] + 1) * X1_Jac(0,3) -2 * (v[3] * v[2] + v[7] * v[6] + 1) *X1_Jac(0,2);
                jacobian_temp(3,4)  = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))*(-v[7])-(-v[3])*(v[4]*X1_Jac(0,1)-v[5]*X1_Jac(0,1));
                jacobian_temp(3,5)  = (v[0]*X1_Jac(0,0)-v[1]*X1_Jac(0,1))*(-1)-(-v[3])*(X1_Jac(0,0)-X1_Jac(0,1));
                jacobian_temp(3,6)  = (v[4]*X1_Jac(0,0)-v[5]*X1_Jac(0,1))*(-1)-(-v[7])*(X1_Jac(0,0)-X1_Jac(0,1));
    
                for(int i=0;i<4;i++)
                {
                    for(int j=0;j<7;j++)
                    jacobian_temp_M(i,j)=jacobian_temp(i,j)+JAC(i,j);
                }
    
                Eigen::JacobiSVD<Eigen::MatrixXd> svd_1(jacobian_temp_M, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
                int row_invJav_M = jacobian_temp_M.rows();
                int col_invJav_M= jacobian_temp_M.cols();
                int k_invJav_M =row_invJav_M>=col_invJav_M? col_invJav_M:row_invJav_M;
                Eigen::MatrixXd X_invJav_M = Eigen::MatrixXd::Zero(col_invJav_M,row_invJav_M);
                Eigen::MatrixXd singularValues_inv_invJav_M = svd_1.singularValues();//奇异值
                Eigen::MatrixXd singularValues_inv_mat_invJav_M = Eigen::MatrixXd::Zero(col_invJav_M, row_invJav_M);
                for (long i = 0; i<k_invJav_M; ++i)
                {
                    if (singularValues_inv_invJav_M(i) > pinvtoler)
                        singularValues_inv_invJav_M(i) = 1.0 / singularValues_inv_invJav_M(i);
                    else 
                        singularValues_inv_invJav_M(i) = 0;
                }
                for (long i = 0; i < k_invJav_M; ++i) 
                {
                    singularValues_inv_mat_invJav_M(i, i) = singularValues_inv_invJav_M(i);
                }
                X_invJav_M=(svd_1.matrixV())*(singularValues_inv_mat_invJav_M)*(svd_1.matrixU().transpose());//X=VS+U* 4*7矩阵

                W_Jac = X0_Jac -2*Y0_Jac*X_invJav_M;//优化之后的迭代

                double depth_temp[4];
                for(int i=0;i<4;i++)
                    depth_temp[i]=depth[i];

                for(int i=0;i<4;i++)
                    depth[i]=W_Jac(0,i);

                Y0_Jac(0,0) = (v[0]* v[0]+ v[4]* v[4]+ 1)*depth[0]* depth[0] - 2 * (v[0] * v[1] +v[4] *v[5] +1)*depth[0] *depth[1] + (v[1] * v[1] + v[5] * v[5] + 1)*depth[1] * depth[1] - v[8] * v[8];
                Y0_Jac(0,1)  = (v[0]* v[0]+ v[4]* v[4]+ 1)*depth[0]* depth[0] - 2 * (v[0] * v[2] +v[4] *v[6] +1)*depth[0] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] * depth[2] - v[9] * v[9];
                Y0_Jac(0,2)  = (v[1] *v[1]+ v[5] *v[5]+ 1)*depth[1]* depth[1] - 2 * (v[1] * v[2] +v[5] *v[6] +1)*depth[1] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] *depth[2] - v[10] * v[10];
                Y0_Jac(0,3)  = (v[3]* v[3]+ v[7]* v[7]+ 1)*depth[3]* depth[3] - 2 * (v[3] * v[2] +v[7] *v[6] +1)*depth[3] *depth[2] + (v[2] * v[2] + v[6] * v[6] + 1)*depth[2] *depth[2] - v[11] * v[11];
                Y0_Jac(0,4)  = (v[0]*depth[0]-v[1]*depth[1])*(v[6]*depth[2]-v[7]*depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(v[4]*depth[0]-v[5]*depth[1]);
                Y0_Jac(0,5)  = (v[0]*depth[0]-v[1]*depth[1])*(depth[2]-depth[3])-(v[2]*depth[2]-v[3]*depth[3])*(depth[0]-depth[1]);
                Y0_Jac(0,6)  = (v[4]*depth[0]-v[5]*depth[1])*(depth[2]-depth[3])-(v[6]*depth[2]-v[7]*depth[3])*(depth[0]-depth[1]);
            
                Eigen::JacobiSVD<Eigen::MatrixXd> svd_2(jacobian_temp, Eigen::ComputeFullU | Eigen::ComputeFullV);//M=USV*
                // double  pinvtoler = 1.e-8; //tolerance
                int row_invJav_T = jacobian_temp.rows();
                int col_invJav_T= jacobian_temp.cols();
                int k_invJav_T =row_invJav_T>=col_invJav_T? col_invJav_T:row_invJav_T;
                Eigen::MatrixXd X_invJav_T = Eigen::MatrixXd::Zero(col_invJav_T,row_invJav_T);
                Eigen::MatrixXd singularValues_inv_invJav_T = svd_2.singularValues();//奇异值
                Eigen::MatrixXd singularValues_inv_mat_invJav_T = Eigen::MatrixXd::Zero(col_invJav_T, row_invJav_T);
                for (long i = 0; i<k_invJav_T; ++i) 
                {
                    if (singularValues_inv_invJav_T(i) > pinvtoler)
                        singularValues_inv_invJav_T(i) = 1.0 / singularValues_inv_invJav_T(i);
                    else 
                        singularValues_inv_invJav_T(i) = 0;
                }
                for (long i = 0; i < k_invJav_T; ++i) 
                {
                    singularValues_inv_mat_invJav_T(i, i) = singularValues_inv_invJav_T(i);
                }
                X_invJav_T=(svd_2.matrixV())*(singularValues_inv_mat_invJav_T)*(svd_2.matrixU().transpose());//X=VS+U* 4*7矩阵
            
                Z_Jac=W_Jac-Y0_Jac*X_invJav_T;

                for(int i=0;i<4;i++)     
                {
                    depth[i]=Z_Jac(0,i);
                    X0_Jac(0,i)=Z_Jac(0,i);
                    x_start[i]=Z_Jac(0,i);   
                }
                
                if(  ((depth_temp[0]-Z_Jac(0,0))+ (depth_temp[1]-Z_Jac(0,1))+ (depth_temp[2]-Z_Jac(0,2))+ (depth_temp[3]-Z_Jac(0,3)))  < Epsilon )
                {
                    break;
                }
            } while (iter<Max);

            //得到四个角点的深度信息
            depth[0]=fabs(depth[0]);
            depth[1]=fabs(depth[1]);
            depth[2]=fabs(depth[2]);
            depth[3]=fabs(depth[3]);
            std::cout<<"深度：P1 " << depth[0] << " P2 " << depth[1] << " P3 " << depth[2] << " P4 " << depth[3] << "cm" <<std::endl;

            //X--与u同向--右方，Y--与v同向--下方
            float X1_camera=trans_P1d[0]*depth[0];
            float Y1_camera=trans_P1d[1]*depth[0];
            float X2_camera=trans_P2d[0]*depth[1];
            float Y2_camera=trans_P2d[1]*depth[1];
            float X3_camera=trans_P3d[0]*depth[2];
            float Y3_camera=trans_P3d[1]*depth[2];
            float X4_camera=trans_P4d[0]*depth[3];
            float Y4_camera=trans_P4d[1]*depth[3];

            //相机坐标
            Pc_central << (depth[1]+depth[2]+depth[0]+depth[3])/4, //正前方
                                        -(X1_camera+X4_camera+X2_camera+X3_camera)/4,//左方
                                        -(Y2_camera+Y3_camera+Y1_camera+Y4_camera)/4;//正上方
            std::cout << "相机坐标Pc:  (前" << Pc_central[0] << ",  左" << Pc_central[1] << ",  上" << Pc_central[2] << ")cm" << endl;  

            ////////////////////////////////////////////////////////////////////////////////检测数字//////////////////////////////////////////////////////////////////////////////////
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

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(2, 2));
            cv::erode(image_threshold_white, image_erode, kernel);
            cv::dilate(image_erode, image_dilate, kernel);
            cv::imshow(NumberDetection, image_dilate);

            /////////////////////////////////////////////////////////////////////////////////识别数字////////////////////////////////////////////////////////////////////////////////////
            Mat num_ROI;

            int min_x = min({min_Point1.x, min_Point2.x, min_Point3.x, min_Point4.x});
            int min_y = min({min_Point1.y, min_Point2.y, min_Point3.y, min_Point4.y});
            int max_x = max({min_Point1.x, min_Point2.x, min_Point3.x, min_Point4.x});
            int max_y = max({min_Point1.y, min_Point2.y, min_Point3.y, min_Point4.y});

            int rect_wid = max_x - min_x;
            int rect_hei = max_y - min_y;

            //只保留边框周边区域作为ROI区域，屏蔽其他区域
            if((min_x - 2) > 0 && (min_y - 2 ) > 0 && rect_wid > 0 && rect_hei > 0 && (max_x + 2) < image_raw.cols && (max_y + 2) < image_raw.rows){
                Rect roi_rect(min_x - 2, min_y - 2, rect_wid + 4, rect_hei + 4);
                mask = Mat::zeros(image_dilate.size(), CV_8UC1);//非ROI区域
                mask(roi_rect).setTo(255);//ROI区域
                // cv::imshow("mask", mask);
                bitwise_and(image_dilate, mask, image_roi);//原图与mask图进行与运算
            }
            else{
                image_roi = image_dilate;
            }
            // cv::imshow("ROI", image_roi);

            //在上图中寻找轮廓
            cv::findContours(image_roi, contours, hierarchy, RETR_EXTERNAL, cv::CHAIN_APPROX_NONE), Point(0,0);

            //滤除掉面积较小和贴近边缘的轮廓
            if(!contours.empty()){
                numContours.clear();
                for(int i = 0; i < contours.size(); i++){
                    if(contourArea(contours[i]) / width*height < 0.005){
                        continue;
                    }
                    Rect numBox = boundingRect(contours[i]);
                    if(numBox.x < 2 || numBox.y < 2 || (numBox.x + numBox.width + 2) > image_raw.cols || (numBox.y + numBox.height + 2) > image_raw.rows){
                        continue;
                    }
                    numContours.push_back(contours[i]);
                }
            }

            //取面积最大的轮廓
            numSquares.clear();
            if(!numContours.empty()){
                //计算每个轮廓的面积
                std::list<int> len_list2;
                for (int i = 0; i < numContours.size(); i++){
                    len_list2.push_back(cv::contourArea(numContours[i]));
                }
                //取面积最大的轮廓(外轮廓)
                if (len_list2.size() > 0){
                    numSquares.push_back(numContours[std::distance(len_list2.begin(), max_element(len_list2.begin(), len_list2.end()))]);
                }
                Rect  NumBoundingBox = boundingRect(numSquares[0]);
                num_ROI = image_dilate(NumBoundingBox);
                // imshow("num_ROI", num_ROI);
                resize(num_ROI, num_ROI, Size(300, 300));
                cv::imwrite("/home/lxl/catkin_ws/src/MonoCamera/roi.jpg", num_ROI);
            }

            //
            //*数字识别方法１——模板匹配
            // num_ROI = imread("/home/lxl/catkin_ws/src/MonoCamera/roi.jpg");
            // if(!numSquares.empty()){
            //     int index = temple_image(num_ROI);
            //     if(index >= 0 && index <= 29){
            //         match_number = 0;
            //     }
            //     else if(index >= 30 && index <= 51){
            //         match_number = 1;
            //     }
            //     else if(index >= 52 && index <= 119){
            //         match_number = 2;
            //     }
            //     else if(index >= 120 && index <= 158){
            //         match_number = 3;
            //     }
            // }
            //

            //*数字识别方法２——轮廓匹配
            num_ROI = imread("/home/lxl/catkin_ws/src/MonoCamera/roi.jpg", IMREAD_GRAYSCALE);
            vector<vector<Point> > num_contour;
            vector<Vec4i> hiera;
            int index;
            findContours(num_ROI, num_contour, hiera, RETR_CCOMP, CHAIN_APPROX_NONE, Point(0, 0));
            if(!numSquares.empty()){
                double hu = 1.0;
                double tmpHu = 0.0;
                for(int i = 0; i < conTemp; i++){
                    tmpHu = matchShapes(mContoursTemp[i], numContours[0], CV_CONTOURS_MATCH_I1, 0);
                    if(tmpHu < hu){
                        hu = tmpHu;
                        index = i;
                    }
                }
                if(index >= 0 && index <= 35){
                    match_number = 0;
                }
                else if(index >= 36 && index <= 68){
                    match_number = 1;
                }
                else if(index >= 69 && index <= 100){
                    match_number = 2;
                }
                else if(index >= 101 && index <= 135){
                    match_number = 3;
                }
            }
            
            cout << "目标数字：" << match_number << endl;
            cout << endl;
            drawNumber(image_raw_copy, match_number);
            imshow("检测结果", image_raw_copy);
        }
        else{
            isDetected = false;
            cout << "!!! 未检测到目标 !!!" << endl;
            cout << endl;
        }
        waitKey(1);

        //初始化消息
        MonoCamera::object obj_msg;
        obj_msg.isDetected = isDetected;
        obj_msg.object_number = match_number;
        obj_msg.object_position.x = Pc_central[0];
        obj_msg.object_position.y = Pc_central[1];
        obj_msg.object_position.z = Pc_central[2];
        //发布消息
        pub.publish(obj_msg);
        loop_rate.sleep();

        //保存数据到当前路径下的txt文件中
        writeData();

        //保存视频
        writer << image_raw;
    }
    cap.release();
    return 0;
}